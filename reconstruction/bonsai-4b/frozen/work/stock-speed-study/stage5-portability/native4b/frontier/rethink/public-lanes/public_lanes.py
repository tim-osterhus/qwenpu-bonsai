"""Place existing guest state with phase-qualified transport and observer reads."""

from collections import Counter, defaultdict, deque
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spill-layout"))
import spill_layout as spill
import opcode_context

ret, full, compact, grid, live, s = (
    spill.ret, spill.full, spill.compact, spill.grid, spill.live, spill.s
)
TERMINAL = (s.HALT, s.FAULT)
RETIREMENT = (s.COMMIT, *TERMINAL)
WIDTH = 2560
PROBE = full.HERE.parent / "story-comparison" / "probe_story.cpp"
OBSERVER = full.HERE / "full9b.observer.txt"


def contract(ids):
    """Read the checkpoint's observer ABI and enumerate scalar fields from the runner."""
    version, *rows = OBSERVER.read_text().splitlines()
    assert version == "QPVXOBSERVER2"

    def unpack(row):
        count, *values = map(int, row.split())
        assert count == len(values)
        return tuple(values)

    labels = ("vmx", "vsum", "varg", "vbest", "vbest_valid", "wrom_index", "tile_extra_0", "tile_extra_1")
    observed = {label: unpack(row) for label, row in zip(labels, rows[:-1], strict=True) if label != "vbest_valid"}
    valid, tile_token = map(int, rows[4].split())
    observed["vbest_valid"] = (valid,)
    token_names = {value: name for name, value in ids.items()}
    read_tokens = tuple(token_names[token] for token in unpack(rows[8]))
    fields = {
        "write_address": s.bits(s.WRITE_ADDR, 33), "write_word": s.bits(s.WRITE_WORD),
        "fresh_handle": s.bits(s.HANDLE_IN, s.HANDLE_BITS), "entry": s.bits(s.ENTRY),
        "read_address": s.bits(s.READ_ADDR, 33), "read_marker": (s.READ_MARK,),
        "old_handle": s.bits(s.OLD_HANDLE, s.HANDLE_BITS), "old_present": (s.OLD_HIT,),
        "pc": s.bits(s.PC), "word": s.bits(s.RAW), "next_pc": s.bits(s.NEXT_PC),
        "cause": s.bits(s.CAUSE, 3), "rd_valid": (s.RD_VALID,), "rd_index": s.bits(s.RD, 5),
        "rd_value": s.bits(s.RD_WORD), "bus_valid": (s.BUS_VALID,), "bus_write": (s.BUS_WRITE,),
        "bus_width": s.bits(s.BUS_WIDTH, 2), "bus_address": s.bits(s.BUS_ADDR), "bus_word": s.bits(s.BUS_WORD),
        "branch_valid": (s.BRANCH_VALID,), "branch_taken": (s.TAKEN,), "putchar_valid": (s.PUTCHAR_FLAG,),
        "exit_code": s.bits(s.EXIT_CODE), "readback_value": s.bits(s.READBACK),
        "readback_handle": s.bits(s.READBACK + 32, s.HANDLE_BITS), "readback_hit": (s.READBACK + 56,),
        **observed,
    }
    source = PROBE.read_text()
    for constant, label in {
        "write_address": "write_address", "value_lane": "write_word", "fresh_handle": "fresh_handle",
        "entry_lane": "entry", "read_address": "read_address", "read_mark": "read_marker",
        "old_handle": "old_handle", "old_present": "old_present", "pc_lane": "pc", "word_lane": "word",
        "next_pc_lane": "next_pc", "cause_lane": "cause", "rd_valid": "rd_valid", "rd_index": "rd_index",
        "rd_value": "rd_value", "bus_valid": "bus_valid", "bus_write": "bus_write", "bus_width": "bus_width",
        "bus_address": "bus_address", "bus_word": "bus_word", "branch_valid": "branch_valid",
        "branch_taken": "branch_taken", "putchar_valid": "putchar_valid", "exit_code": "exit_code",
        "readback": "readback_value",
    }.items():
        assert f"{constant} = {fields[label][0]}" in source
    assert f"handle_bits = {s.HANDLE_BITS}" in source
    assert "feedback[bus_width] == 0.f && feedback[bus_width + 1] == 0.f" in source
    assert "hidden[read_mark] > 0.25f" in source and "feedback[putchar_valid] > 0.f" in source
    assert "emit && input >= 313 && input <= 316" in source
    return fields, read_tokens, token_names[tile_token], tuple(token_names[t] for t in range(313, 317))


def guarded_values(c, states, edges, names, guards):
    """Prove guards false only by an initialized literal or its exact refresh."""
    # The sets are sign possibilities. Nothing is removed on an unresolved
    # computed guard or a branch whose controlling value is not this guard.
    incoming = [[set() for _ in guards] for _ in states]
    for i, (phase, _) in enumerate(states):
        if names[phase] in ret.ROOTS:
            for slot in incoming[i]:
                slot.add(0)
    possibilities = {-1, 0, 1}
    outputs = [[set() for _ in guards] for _ in states]
    changed = True
    while changed:
        changed = False
        for i, (phase, _) in enumerate(states):
            name = names[phase]
            for j, lane in enumerate(guards):
                before = incoming[i][j]
                if not before:
                    continue
                if name in (s.SEED_MISS, s.SEED_WORD):
                    default = ()
                elif name == s.START:
                    default = s.constant(-1)
                else:
                    default = s.buffer(s.WORK + lane)
                effect = c.effects[name].get(lane, default)
                if effect == ():
                    after = {0}
                elif effect in (s.constant(1), s.constant(-1)):
                    after = {int(effect[0][0])}
                elif effect in (s.buffer(s.WORK + lane), s.buffer(s.WORK + lane, -1)):
                    after = {int(effect[0][0]) * val for val in before}
                else:
                    after = possibilities
                outputs[i][j] = after
                for target in edges[i]:
                    older = incoming[target][j]
                    flag = c.routes[name][names[states[target][0]]]
                    routed = (after & {1 if flag > 0 else -1}) if flag in (lane, -lane) else after
                    # Keep a conservative value if an impossible abstract edge
                    # is still in the branch graph.
                    routed = routed or after
                    if not routed <= older:
                        older.update(routed)
                        changed = True
    assert all(all(i) for i in incoming)
    return outputs


def host_needs(c, ids, states, edges, fields, read_tokens, tile, range_guard):
    """Model what each edge's feedback must expose to the launcher."""
    names = tuple(ids)
    watched = ("rd_valid", "bus_valid", "branch_valid", "putchar_valid", "old_present", "vbest_valid")
    guards = tuple(fields[name][0] for name in watched)
    outputs = guarded_values(c, states, edges, names, guards)
    before_events, after_events = Counter(), Counter()
    guard_events = defaultdict(Counter)
    before_by_action, after_by_action = defaultdict(set), defaultdict(set)
    fault_producers = {}

    def guarded_for(i, target, guard, field_names):
        j = watched.index(guard)
        values = outputs[i][j]
        source = names[states[i][0]]
        flag = c.routes[source][target]
        if flag in (guards[j], -guards[j]):
            values = values & {1 if flag > 0 else -1} or values
        needed = 1 in values
        guard_events[guard]["possible_positive" if needed else "proven_nonpositive"] += 1
        return set(field_names) if needed else set()

    def pre_fields(i, target):
        wanted = {"read_marker"}  # Logged for all actions, including STOP and terminal.
        if target == s.FETCH or target in read_tokens or target == tile:
            wanted.add("read_address")
        if target in (tile, *range_guard):
            wanted.add("wrom_index")
        if target in (s.STORE, s.REGISTER):
            wanted.update(("write_address", "write_word"))
        if target == s.RETIRE:
            wanted.add("old_present")
            wanted |= guarded_for(i, target, "old_present", ("old_handle",))
        if target in RETIREMENT:
            wanted.update(("pc", "word", "next_pc", "rd_valid", "bus_valid", "branch_valid", "putchar_valid",
                           "vmx", "vsum", "varg", "vbest_valid"))
            wanted |= guarded_for(i, target, "rd_valid", ("rd_index", "rd_value"))
            wanted |= guarded_for(i, target, "bus_valid", ("bus_write", "bus_width", "bus_address", "bus_word"))
            wanted |= guarded_for(i, target, "branch_valid", ("branch_taken",))
            wanted |= guarded_for(i, target, "vbest_valid", ("vbest",))
            if target == s.HALT:
                wanted.add("exit_code")
            else:
                output = guarded_for(i, target, "putchar_valid", ("output_byte",))
                wanted |= output
            if target == s.FAULT:
                wanted.add("cause")
                fault_producers[names[states[i][0]]] = sorted(wanted)
        return wanted

    masks = []
    fields = {**fields, "output_byte": fields["exit_code"][:9]}
    for i, (phase, _) in enumerate(states):
        source = names[phase]
        after = {"readback_value", "readback_handle", "readback_hit"} if source in read_tokens else set()
        if source == tile:
            after.update(("tile_extra_0", "tile_extra_1"))
        after_by_action[source] |= after
        after_events.update(after)
        mask = live.mask(lane for label in after for lane in fields[label])
        for t in edges[i]:
            target = names[states[t][0]]
            before = pre_fields(i, target)
            before_by_action[target] |= before
            before_events.update(before)
            mask |= live.mask(lane for label in before for lane in fields[label])
        masks.append(mask)
    assert set(fault_producers) == {name for name in ids if name != s.FAULT and s.FAULT in c.routes[name]}
    report = dict(before_fields_by_action={key: sorted(value) for key, value in before_by_action.items() if value},
                  after_fields_by_action={key: sorted(value) for key, value in after_by_action.items() if value},
                  before_field_pred_edges=dict(before_events), after_field_states=dict(after_events),
                  guard_outcomes={key: dict(val) for key, val in guard_events.items()}, fault_producers=fault_producers)
    return masks, report


def close(states, edges, deps, base, actual, host):
    """Solve liveness with both post-decode reads and the next action's host reads."""
    parents = [[] for _ in states]
    for j, targets in enumerate(edges):
        for target in targets:
            parents[target].append(j)

    def solve(forced):
        ins, outs = [0] * len(states), [0] * len(states)
        queue, pending = deque(range(len(states))), set(range(len(states)))
        while queue:
            i = queue.popleft()
            pending.remove(i)
            phase, _ = states[i]
            outs[i] = after = host[i] | live.union_bits(ins[j] for j in edges[i])
            explicit, defaults = deps[phase]
            emits = after if forced is None else forced[phase]
            value = base[phase] | (emits & defaults)
            value |= live.union_bits(read for bit, read in explicit if bit & emits)
            if ins[i] != value:
                assert ins[i] & ~value == 0
                ins[i] = value
                for parent in parents[i]:
                    if parent not in pending:
                        queue.append(parent)
                        pending.add(parent)
        return ins, outs

    ideal = solve(None)
    ins, outs = ideal
    keep = [0] * len(actual)
    passes = 0
    while True:
        by_phase = ret.phase_union(states, outs, len(actual))
        next_keep = [old | (need & written) for old, need, written in zip(keep, by_phase, actual, strict=True)]
        if next_keep == keep:
            return ideal, ins, outs, keep, passes
        keep = next_keep
        ins, outs = solve(keep)
        passes += 1


def observed_pool(c, ports, ingress, raw, qkv, old_private):
    dot_ports = {lane for value in ports.values() for lane in ((value,) if isinstance(value, int) else value)}
    attention = set().union(*qkv.values())
    candidates = (raw & ingress) - dot_ports - attention
    assert candidates <= set(c.persistent) and old_private <= candidates
    assert len(candidates - old_private) == 402
    return candidates, attention, dot_ports


def emitted_witness(states, edges, names, kept, ins, outs, protected_outs, dst, defaults, actual):
    """Find one real phase emission with distinct projected reads at its next action."""
    phase = max(range(len(kept)), key=lambda p: kept[p].bit_count())
    for i, (p, ctx) in enumerate(states):
        if p != phase:
            continue
        for target in edges[i]:
            q, next_ctx = states[target]
            if kept[p] == ins[target]:
                mask = kept[p]
                assert mask & actual[p] == mask and mask & outs[i] == mask
                assert mask == protected_outs[i]
                work_default_reads = ins[target] & defaults[q] & kept[q]
                work_explicit_reads = ins[target] & live.union_bits(
                    reads for lane, reads in dst[q].items() if (1 << lane) & kept[q])
                return dict(emitting_action=names[p], emitting_context=ctx, next_action=names[q], next_context=next_ctx,
                            real_retained_outputs=mask.bit_count(), distinct_projected_work_inputs=ins[target].bit_count(),
                            real_explicit_outputs=(mask & live.mask(dst[p])).bit_count(),
                            real_refresh_outputs=(mask & defaults[p]).bit_count(),
                            projected_default_reads=work_default_reads.bit_count(),
                            projected_explicit_reads=work_explicit_reads.bit_count(),
                            all_lanes=list(live.members(mask)))
    raise AssertionError("Could not find a same-state real emission witness")


def matching_trace(ids, callers, witness):
    """Find the same contextual transition in an existing completed stock trace."""
    key = (ids[witness["emitting_action"]], ids[witness["next_action"]])
    assert key == (ids["ADD_INIT"], ids["PREFIX_1"])
    valid_callers = {ids[p] for p, mode in callers.items() if mode == witness["emitting_context"]}
    saved = full.HERE.parent / "validation"
    for fixture in ("scalar", "vector-short", "attention"):
        payload = (saved / fixture / "guest.json").read_bytes()
        for index, run in enumerate(json.loads(payload)["runs"]):
            rows = run["actions"]
            for prior, current, after in zip(rows, rows[1:], rows[2:]):
                if prior["action"] in valid_callers and (current["action"], after["action"]) == key:
                    return dict(fixture=fixture, run=index, receipt_sha256=sha256(payload).hexdigest(),
                                caller=tuple(ids)[prior["action"]], emitting_position=current["position"],
                                projected_input_position=after["position"])
    raise AssertionError("No corroborating contextual transition in short stock traces")


def audit():
    prior, original = ret.main()
    baseline = prior["fixed_phase_pruning"]
    assert baseline["fixed_map_colors_under_previous_no_duplicate_emits_remap"] == 555
    assert baseline["candidate_phase_writes_kept"] == 212485
    print("Verified 555-color / 212485-write original control", file=sys.stderr, flush=True)
    c, ids, (states, declared_edges), _, _, _, _, _, old_private, ingress, raw, scratch, qkv, phases, g, u, d = original
    names = tuple(ids)
    terminals = {ids[phase] for phase in TERMINAL}
    edges = [() if phase in terminals else links for (phase, _), links in zip(states, declared_edges, strict=True)]
    _, (_, _, _, _, _, source_rows), ports, gates_dot, _ = full.build()
    dot_start = source_rows - len(compact.structured.dot_plane.compile_ff(False, True).g) - 4 * len(gates_dot)
    active = spill.reachable(c, ids)
    scores = {ids[name]: code for code, name in enumerate(active)}
    tensors, groups, selector = spill.classify_grid(c, ids, scores, ports, gates_dot, dot_start, g, u, ())
    writer, cases = spill.next_fragments(c, ids, active, 9)
    assert selector.input_roles == 67 and cases > 0
    candidates, attention, dot_ports = observed_pool(c, ports, ingress, raw, qkv, old_private)
    dst, defaults, base, recomputed_phases, _, _ = live.summarize_rows(c, ids, g, u, d, candidates, scratch)
    assert recomputed_phases == phases
    for (terms, _, wanted), _outputs in writer:
        reads = live.mask(col-s.WORK for col, val in terms if val and col-s.WORK in candidates)
        for code in live.members(wanted):
            base[ids[active[code]]] |= reads
    fixed_floor = len(raw-candidates) + len(ingress-candidates) + len(scratch) + len(c.dot_payload_ingress) + selector.input_roles + 32
    assert fixed_floor == 1061
    fields, read_tokens, tile, range_guard = contract(ids)
    host_masks, host_report = host_needs(c, ids, states, edges, fields, read_tokens, tile, range_guard)
    assert len(active) == 510 and tile not in active and set(read_tokens) <= set(active)
    assert {ids[phase] for phase in read_tokens} == {ids[phase] for phase in set(full.ORDINARY) | c.vx.reads}
    host_report["guard_real_sparse_write_checks"] = opcode_context.verify_sparse_writers(
        c, ids, {s.RD_VALID, s.BUS_VALID, s.BRANCH_VALID, s.PUTCHAR_FLAG, s.OLD_HIT, fields["vbest_valid"][0]},
        phases, g, u, d)
    host_report["bus_width_real_sparse_write_checks"] = opcode_context.verify_sparse_writers(
        c, ids, set(fields["bus_width"]), phases, g, u, d)
    host_report["ingress_and_modes"] = {
        "seed_miss": {"host_input": "all-zero", "decode_output": False, "retained": True},
        "seed_word": {"host_input_fields": ["write_address", "write_word", "fresh_handle", "WROM payload and reference"],
                      "decode_output": False, "retained": True},
        "start": {"host_input_fields": ["entry"], "all_other_raw_inputs": "zero"},
        "diagnostic_FETCH": {"host_input_fields": ["read_address", "read_marker"], "all_other_raw_inputs": "zero",
                             "postdecode": ["readback_value", "readback_hit", "readback_handle if hit", "next action"]},
        "write_decode": {"injected_after_predecode_logging": ["fresh_handle"]},
        "observer_disabled": {"read_condition": "read_marker > 0.25", "all_query_and_readback_ports_fixed": True},
        "legacy_tile": {"action": tile, "token": ids[tile], "reachable": False,
                        "before": ["read_address (32 bits)", "wrom_index"], "after": ["tile_extra_0", "tile_extra_1"]},
        "interleaved_handoff": {"source": "visible byte stream or HALT tracker summary / exit code", "adds_guest_lanes": False},
        "kv_witness": {"timing": "pre-FFN", "source": "physical K/V cache", "needs_unmodified_ports": True},
        "feedback_hash_dump": {"semantic_lanes_added": False},
    }
    assert set(c.vx.fields["vbest_valid"]) == set(fields["vbest_valid"])
    assert set(fields["read_marker"] + fields["read_address"] + fields["write_address"] + fields["write_word"]
               + fields["wrom_index"]) <= attention
    assert set(fields["readback_value"] + fields["readback_handle"] + fields["readback_hit"]) <= raw - ingress
    deps = ret.active_deps(dst, defaults)
    actual = ret.phase_writes(d, phases, candidates, len(ids))
    candidate_mask = live.mask(candidates)
    host_masks = [x & candidate_mask for x in host_masks]
    cause_bits = live.mask(fields["cause"])
    fault_writers = [ids[phase] for phase in host_report["fault_producers"]]
    assert len(fault_writers) == 8 and all(actual[p] & cause_bits == cause_bits for p in fault_writers)
    results = {}
    for label, needs in (("optimistic_without_any_host_reads", [0] * len(states)), ("guarded_host", host_masks)):
        ideal, ins, outs, kept, passes = close(states, edges, deps, base, actual, needs)
        protected_outs = [out | kept[p] for out, (p, _) in zip(outs, states, strict=True)]
        colors = live.color(ins + protected_outs)
        assert all(len({colors[lane] for lane in live.members(mask)}) == mask.bit_count() for mask in ins + protected_outs)
        ncolor = len(set(colors.values()))
        print(f"{label}: {ncolor} colors, largest output {max(map(int.bit_count, protected_outs))}",
              file=sys.stderr, flush=True)
        result = dict(closure_passes=passes, kept_phase_writes=sum(map(int.bit_count, kept)),
                      original_explicit_phase_writes=sum(map(len, dst)),
                      kept_explicit_phase_writes=sum((mask & live.mask(effects)).bit_count()
                                                     for mask, effects in zip(kept, dst, strict=True)),
                      kept_defaults_or_constant_writes=sum((mask & ~live.mask(effects)).bit_count()
                                                          for mask, effects in zip(kept, dst, strict=True)),
                      distinct_colors=ncolor, distinct_logical_pairs=len(colors), active_host_lanes=live.union_bits(needs).bit_count(),
                      fixed_input_peak=ret.summary(states, ins, outs, names),
                      ideal_input_peak=ret.summary(states, *ideal, names))
        result["actual_emission_and_next_work_witness"] = emitted_witness(
            states, edges, names, kept, ins, outs, protected_outs, dst, defaults, actual)
        witness = result["actual_emission_and_next_work_witness"]
        witness["previously_private_pairs"] = len(set(witness["all_lanes"]) & old_private)
        witness["newly_unpinned_pairs"] = len(set(witness["all_lanes"]) - old_private)
        witness["matching_existing_stock_trace"] = matching_trace(ids, prior["callers"], witness)
        result["fault_producer_cause_retention"] = {names[p]: list(live.members(kept[p] & cause_bits)) for p in fault_writers}
        if label == "guarded_host":
            assert all(kept[p] & cause_bits == cause_bits for p in fault_writers)
            no_cause = [need & ~cause_bits for need in needs]
            _, _, _, without_public_cause, _ = close(states, edges, deps, base, actual, no_cause)
            assert all(without_public_cause[p] & cause_bits == 0 for p in fault_writers)
            result["fault_cause_omission_negative_control"] = len(fault_writers)
        # A raw lane read as a literal zero must still block other retained outputs
        # from claiming its coordinate. This includes exact-zero BUS_WIDTH cases.
        result["required_raw_zero_slots_without_an_emitted_row"] = sum(
            (out & ~actual[p]).bit_count() for out, (p, _) in zip(outs, states, strict=True))
        result["bus_width_read_edges_with_zero_slot_reserved"] = sum(
            bool(need & live.mask(fields["bus_width"]) & ~actual[p]) for need, (p, _) in zip(needs, states, strict=True))
        result["projected_input_and_raw_output_conflicts_checked"] = sum(
            mask.bit_count() * (mask.bit_count()-1) // 2 for mask in ins + protected_outs)
        for group, supports in (("fixed", (ins, protected_outs)), ("ideal", ideal)):
            lower = ret.verified_fixed_map_clique(supports[0] + supports[1])
            ret.certify_clique(lower, states, *supports, names)
            result[f"{group}_verified_clique"] = lower
        result["width_budget"] = {}
        for label_width, saving in (("full", 0), ("half_tested_fixture", 128), ("quarter_conditional", 192)):
            floor = fixed_floor-saving
            result["width_budget"][label_width] = dict(fixed_roles=floor, color_budget=(WIDTH-floor)//2,
                ideal_clique_min_lanes=floor+2*result["ideal_verified_clique"]["slots"],
                fixed_clique_min_lanes=floor+2*result["fixed_verified_clique"]["slots"],
                checked_colors_lanes=floor+2*ncolor)
        results[label] = result
    return dict(control={"colors": 555, "kept_phase_writes": 212485}, candidates=len(candidates), newly_eligible=len(candidates-old_private),
                inventory=dict(raw=len(raw), ingress=len(ingress), scratch=len(scratch), attention_pins=len(attention & raw & ingress),
                               dot_pins_outside_attention=len((dot_ports-attention) & raw & ingress)),
                source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__), Path(ret.__file__), Path(spill.__file__), Path(full.__file__), Path(live.__file__),
                                Path(opcode_context.__file__), PROBE, OBSERVER)},
                host=host_report, guarded_actions=dict(observer_read=list(read_tokens), tile=tile, predecode_wrom=range_guard),
                graph_states=len(states), graph_edges=sum(map(len, edges)), action_tokens=ids,
                logical_fields=fields, logical_fields_requiring_static_abi_map=sorted(
                    label for label, lanes in fields.items() if set(lanes) & candidates), results=results)


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
