"""Symbolic disjoint grid selectors; no models, runtime calls, or GPU work.

Run with ``python -B grid_census.py > grid-census.json``.  The public
``GridSelector`` and ``build_grid`` helpers are also used by bounded probes.
This census does not prove numerical exactness or a physical lane allocation.
"""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FRONTIER = HERE.parents[1]
ROOT = HERE.parents[6]
sys.path[:0] = [str(FRONTIER / "ffn"),
               str(ROOT / "work/prism-upstream/build-source/gguf-py")]
import compact_ffn_audit as compact

# Abstract, disjoint control locations, deliberately outside the old witness.
# A runtime fixture must supply its own physical placement for these roles.
GROUP = 11000
MEMBER = 11100
REFERENCE = 11200


class GridSelector:
    """Natural-order action IDs factored into group/member one-hot inputs.

    Each action has exactly four nonzero input coordinates, all +0.5:
    its protected class, its group, its member, and REFERENCE.  The reference
    is only for constant unit embedding energy.  The existing nine-sign
    next-action output and Q4 score head are retained.
    """

    def __init__(self, class_masks, columns=32, count=510):
        self.class_masks = tuple(class_masks)
        self.columns = columns
        self.count = count
        self.groups = (count + columns - 1) // columns
        self.input_roles = len(self.class_masks) + self.groups + columns + 1
        self._plans = {}

    def grid_cover(self, wanted):
        equal_members = defaultdict(list)
        for group in range(self.groups):
            members = tuple(member for member in range(self.columns)
                            if (code := group * self.columns + member) < self.count
                            and (wanted >> code) & 1)
            if members:
                equal_members[members].append(group)
        return tuple(("grid", (tuple(groups), members))
                     for members, groups in equal_members.items())

    def cover(self, wanted, policy="pure"):
        key = wanted, policy
        if key in self._plans:
            return self._plans[key]
        pure = self.grid_cover(wanted)
        classes = tuple(i for i, mask in enumerate(self.class_masks)
                        if mask & wanted == mask)
        covered = 0
        for i in classes:
            covered |= self.class_masks[i]
        mixed = ((("class", classes),) if classes else ()) + self.grid_cover(wanted ^ covered)
        if policy == "pure":
            result = pure
        elif policy == "class_first":
            result = mixed
        elif policy == "shorter":
            result = pure if len(pure) < len(mixed) else mixed
        else:
            raise ValueError(policy)
        union = 0
        for control in result:
            selected = self.evaluate(control)
            assert selected and not (selected & union)
            union |= selected
        assert union == wanted
        self._plans[key] = result
        return result

    def evaluate(self, control):
        kind, spec = control
        if kind == "class":
            return sum(self.class_masks[i] for i in spec)
        groups, members = spec
        return sum(1 << (group * self.columns + member)
                   for group in groups for member in members
                   if group * self.columns + member < self.count)

    def coefficients(self, control, bias, penalty):
        kind, spec = control
        if kind == "class":
            return (tuple((compact.CLASS + i, 2 * bias) for i in spec),
                    tuple((compact.CLASS + i, 32) for i in spec))
        groups, members = spec
        # Exactly one group input is nonzero.  Increasing an off coefficient
        # does not change the arithmetic of a selected gate.
        gate = tuple((GROUP + i, 2 * bias if i in groups else -2 * penalty)
                     for i in range(self.groups))
        up = tuple((MEMBER + i, 32) for i in members)
        return gate, up

    def embedding(self, code):
        assert 0 <= code < self.count
        cls = next(i for i, mask in enumerate(self.class_masks) if mask >> code & 1)
        return ((compact.CLASS + cls, 0.5),
                (GROUP + code // self.columns, 0.5),
                (MEMBER + code % self.columns, 0.5),
                (REFERENCE, 0.5))


def penalty_for(terms, bias, off_scale=1):
    assert off_scale > 0 and off_scale & (off_scale - 1) == 0
    bound = 8 * sum(abs(value) for _, value in terms) + 4096 + abs(bias)
    return (1 << int(bound).bit_length()) * off_scale


def build_grid(columns=32, policy="pure", off_scale=1, source_rows=None):
    """Return rows and exact plans indexed by (data|writer, source ordinal)."""
    source_rows = compact.source() if source_rows is None else source_rows
    circuit, ids, reachable, codes, _, _, _, _, _, _, _, _, _, _ = source_rows
    data, _, dot, masks, input_lane, baseline = compact.remap_source(source_rows)
    writer, transitions, _ = compact.transition_fragments(
        circuit, ids, reachable, masks, input_lane, False)
    selector = GridSelector(masks, columns, len(codes))
    rows = {key: dict(outputs) for key, outputs in dot.items()}
    plans = {}
    fragments = {"data": data, "writer": writer}
    fragment_count = 0
    for kind, values in fragments.items():
        for ordinal, ((terms, bias, wanted), outputs) in enumerate(values):
            plan = selector.cover(wanted, policy)
            plans[kind, ordinal] = plan
            for control in plan:
                gate, up = selector.coefficients(control, bias, penalty_for(terms, bias, off_scale))
                compact.add_row(rows, (terms + gate, up), outputs)
                fragment_count += 1
    for code in range(len(codes)):
        assert sum(value * value for _, value in selector.embedding(code)) == 1
    return dict(rows=rows, selector=selector, plans=plans, fragments=fragments,
                baseline=baseline, transitions=transitions, fragment_count=fragment_count,
                source_rows=source_rows)


def storage(rows):
    shared = (max(rows, 9216) + 511) // 512 * 512
    padding = (shared - 9216) * 65280
    circuit = shared * 2560 * 8  # F32 gate, BF16 up, F16 down.
    return dict(shared_h512_rows=shared,
                language_padding_rows_per_matrix=shared - 9216,
                language_padding_bytes=padding, mtp_ffn_bytes=circuit,
                language_padding_plus_mtp_ffn_bytes=padding + circuit,
                device_proxy_mib=round(7274.35 + (shared - 53760) / 512 * 41.875, 3))


def serial_adder_counterweight(ids):
    path = ROOT / "work/stock-speed-study/stage5-portability/full9b/story-comparison/9b/story.json"
    capture = json.loads(path.read_text())
    run = capture["runs"][0]
    counts = Counter(row["action"] for row in run["actions"])
    phases = ("ADD_INIT", "PREFIX_1", "PREFIX_2", "PREFIX_4", "PREFIX_8", "PREFIX_16", "ADD_FINISH")
    phase_visits = {phase: counts[ids[phase]] for phase in phases}
    assert len(set(phase_visits.values())) == 1
    additions = phase_visits["ADD_INIT"]
    added_steps = (32 - len(phases)) * additions
    return dict(scope="Workload counterweight for an unproved replacement; not a runtime prediction.",
                source=str(path.relative_to(ROOT)), source_sha256=sha256(path.read_bytes()).hexdigest(),
                non_seed_decodes=len(run["actions"]), retirements=run["retirement_attempts"],
                existing_core_decodes_per_add=len(phases), phase_visits=phase_visits,
                proposed_minimum_core_decodes_per_add=32,
                extra_decodes_before_routing_or_spills=added_steps,
                projected_non_seed_decodes=len(run["actions"]) + added_steps,
                projected_non_seed_decode_ratio=(len(run["actions"]) + added_steps) / len(run["actions"]),
                warning="X32/P31/G32 comprise 95 logical scratch bits, but only 31 colors are exclusive to the prefix-scratch family; global physical savings require recoloring.")


def main():
    source_rows = compact.source()
    results = []
    for columns in (16, 23, 32):
        for policy in ("pure", "class_first", "shorter"):
            built = build_grid(columns, policy, source_rows=source_rows)
            summary = compact.row_summary(built["rows"])
            selector = built["selector"]
            results.append(dict(columns=columns, groups=selector.groups, policy=policy,
                                input_roles=selector.input_roles,
                                row_fragments=built["fragment_count"],
                                activation_decisions_checked=510 * sum(map(len, built["fragments"].values())),
                                **summary, storage=storage(summary["rows"])))
    best = next(r for r in results if r["columns"] == 32 and r["policy"] == "pure")
    receipt = dict(
        scope="Symbolic sparse census only; no numerical, packed-tile, physical-allocation, full-model or GPU proof.",
        source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(compact.__file__), Path(compact.live.__file__),
            Path(compact.full.__file__), FRONTIER / "hybrid/compiler/hybrid-census.json")},
        source_baseline=built["baseline"],
        map="Dense 0..509 reachable actions in original token order; group=code//columns; member=code%columns.",
        coefficient_rule="Four active +0.5 inputs: protected class, group, member, reference. Gate keeps predicate terms; group coefficient is 2*bias if admitted, else -2*P. Up member coefficient is 32 if admitted, else zero.",
        penalty_rule="P=2**bit_length(8*sum(abs(predicate coefficients))+4096+abs(bias)); bounded probes may increase P by a power of two without changing row count or selected-group coefficients.",
        abstract_coordinates=dict(protected_class=compact.CLASS, group=GROUP, member=MEMBER,
                                  reference=REFERENCE, next_sign=compact.NEXT),
        results=results,
        chosen=dict(columns=32, groups=16, input_roles=67, rows=best["rows"],
                    storage=best["storage"],
                    base_roles_with_27_input_coordinates=2937,
                    added_input_roles=40, split_wrom_saving=128,
                    assumed_binary_color_collapses=246, anchor_roles=2,
                    conditional_roles=2937 + 40 - 128 - 246 + 2,
                    conditional_excess_over_2560=45),
        serial_adder_counterweight=serial_adder_counterweight(source_rows[1]),
        quarter_wrom_alternative=dict(scope="Unproved architectural option; no quarter-ROM fixture was run.",
                    projected_role_saving=192,
                    conditional_roles=2937 + 40 - 192 - 246 + 2,
                    extra_retained_cells=3 * 4608,
                    bounded_prefix_projected_live_cells=55720 + 3 * 4608,
                    overflow_at_65536=55720 + 3 * 4608 - 65536,
                    illustrative_context=73728,
                    bounded_prefix_free_cells_at_73728=73728 - 55720 - 3 * 4608,
                    additional_f16_kv_payload_bytes=(73728 - 65536) * 4096),
        open_gates=["Off-group exact SiLU saturation across admitted amplitudes/work inputs on both stock backends.",
                    "Whole-destination centered zeros and stock DOWN reduction with the final row ordering/shape.",
                    "Next-code writer predicates must satisfy its bipolar, single-winning-route/fallback domain or reproduce the original score resolution.",
                    "All 246 binary color collapses and integrated physical allocation, including PQ2 tile sharing.",
                    "Packed initialization and global energy bound for any fixed-RMS anchors.",
                    "Typed WROM addressing, CUDA FlashAttention, full-story KV capacity and text-state/logit parity.",
                    "Measured full checkpoint device peak; device_proxy_mib is an old weight-delta proxy."])
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
