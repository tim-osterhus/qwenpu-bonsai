"""Check logical emissions and preserved physical observations on every edge."""

from collections import defaultdict
from hashlib import sha256
import json


def proof(details, deps, place, abi, callers, fields, read_tokens, tile, range_guard, grid):
    spill, public, live, ret, s = grid
    c, ids, active, _, _, _, _, _, downs, row_phases, _, _, kept, colors, _, raw, pool, _, _, _ = details
    dst, defaults, base, actual = deps
    names = tuple(ids)
    states, edges = spill.flow(c, ids, callers, {mode: f"AFTER_{mode}" for mode in c.returns})
    edges = [() if names[p] in public.TERMINAL else targets for (p, _), targets in zip(states, edges, strict=True)]
    host, host_report = public.host_needs(c, ids, states, edges, fields, read_tokens, tile, range_guard)
    for field in ("before_field_pred_edges", "after_field_states"):
        host_report[field] = dict(sorted(host_report[field].items()))
    pool_mask = live.mask(pool)
    candidate_host = [mask & pool_mask for mask in host]
    _, ins, outs, keep2, passes = public.close(
        states, edges, ret.active_deps(dst, defaults), base, actual, candidate_host)
    assert keep2 == kept
    fixed_raw = place["fixed_raw"]
    raw_map = fixed_raw | place["raw"]
    host_only = {
        pos for key in ("fresh_handle", "entry", "wrom_signed_nibbles", "wrom_reference", "wrom_norm_anchors")
        for pos in abi["raw_ingress"][key]
    }
    qk_raw = live.mask(fields["write_address"]) | live.mask(fields["write_word"])
    wrom_phase = {"DOT64_CN", "DOT64_CP", "DOT64_FN", "DOT64_FP"}
    write_phase = {s.STORE, s.REGISTER}
    pure_pins = [0] * len(ids)
    for row, supported in zip(downs, row_phases, strict=True):
        mask = live.mask(row.keys() & fixed_raw.keys())
        for token in supported:
            pure_pins[token] |= mask
    reachable = set(ids[name] for name in active)
    readback = live.mask(s.bits(s.READBACK, 57))
    score_reserved = set(place["score_tile"]) - set(range(place["score_start"], place["score_start"] + 9))
    score_written = set(place["score_tile"]) - score_reserved
    host_domain = live.mask(raw)
    phase_rows = {}
    all_scores = []
    for token in sorted(reachable):
        emissions = pure_pins[token] | kept[token] | readback
        physical = {raw_map[lane] for lane in live.members(emissions)}
        assert not kept[token] & ~actual[token]
        assert len(physical) == emissions.bit_count()
        assert not physical & (host_only | set(place["score_tile"]))
        assert physical <= set(place["head_feedback"])
        phase_rows[token] = dict(action=names[token], retained_logical_pair_writes_hex=format(kept[token], "x"),
                                 possible_physical_ffn_and_O_outputs_hex=format(live.mask(physical), "x"),
                                 restored_from_literal_zero_count=0)
        if names[token] not in public.TERMINAL:
            all_scores.append(token)
    assert set(score_written).isdisjoint(raw_map.values())
    assert set(score_reserved).isdisjoint(raw_map.values())
    width_zeros = defaultdict(int)
    width_contexts = defaultdict(int)
    zero_cases = 0
    for i, ((token, mode), successors) in enumerate(zip(states, edges, strict=True)):
        phase = names[token]
        required = (host[i] & host_domain) | outs[i]
        protected = required | kept[token]
        assert host[i] & ~host_domain == 0
        assert len({raw_map[lane] for lane in live.members(protected)}) == protected.bit_count()
        assert len({place["work"][lane] for lane in live.members(ins[i])}) == ins[i].bit_count()
        assert ins[i] & ~live.mask(colors) == 0
        emitted = pure_pins[token] | kept[token] | readback
        zero = required & ~emitted
        same_emitter = live.mask(raw_map[lane] for lane in live.members(emitted))
        assert not same_emitter & live.mask(raw_map[lane] for lane in live.members(zero))
        phase_rows[token]["restored_from_literal_zero_count"] += zero.bit_count()
        zero_cases += zero.bit_count()
        width = live.mask(fields["bus_width"])
        if (host[i] & width) == width:
            width_contexts[phase] += 1
            if (zero & width) == width:
                width_zeros[phase] += 1
        for j in successors:
            target = names[states[j][0]]
            # W queries carry no newly written key or payload, even though
            # stock inserts the scratch row before the attention read.
            if target in wrom_phase:
                assert not pure_pins[token] & qk_raw
            if target in write_phase:
                assert (pure_pins[token] & qk_raw) == qk_raw
    report = dict(context_states=len(states), checked_edges=sum(map(len, edges)), closure_solves=passes,
                  decode_output_phases=len(all_scores), emitted_pair_phase_writes=sum(map(int.bit_count, kept)),
                  protected_implicit_zero_context_slots=zero_cases,
                  bus_width_contexts=sum(width_contexts.values()),
                  bus_width_literal_zero_contexts=dict(sorted(width_zeros.items())), transport_host_edges=host_report)
    # The stock probe special-cases the all-zero pair. The compiled ordinary
    # guarded contexts also preserve its two individually readable slots.
    width_physical = abi["feedback_fields"]["bus_width"]
    assert len(set(width_physical)) == 2 and width_contexts
    report["bus_width_compatibility_controls"] = width_controls(place["width"], width_physical)
    return report, phase_rows


def write_phase_masks(path, phases):
    text = json.dumps(phases, separators=(",", ":")) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return sha256(text.encode("utf-8")).hexdigest()


def width_controls(width, lanes):
    def parse(feedback):
        low, high = [feedback[i] for i in lanes]
        if low == high == 0:
            return 4
        if low == 0 or high == 0:
            return "ambiguous"
        return {0: 4, 1: 1, 2: 2}.get((low > 0) + 2 * (high > 0), "invalid")

    cases = {(0, 0): 4, (-1, -1): 4, (1, -1): 1, (-1, 1): 2, (0, -1): "ambiguous", (1, 1): "invalid"}
    feedback = [0.] * width
    for bits, expected in cases.items():
        for lane, bit in zip(lanes, bits, strict=True):
            feedback[lane] = float(bit)
        assert parse(feedback) == expected
    return {str(bits): expected for bits, expected in cases.items()}
