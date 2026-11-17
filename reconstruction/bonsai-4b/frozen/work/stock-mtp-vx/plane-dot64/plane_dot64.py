"""Isolated stock-Qwen35 dynamic signed int8 x offset-int4 64-lane dot probe.

No Bonsai input is read. Runtime activation gate bits and a fixed three-phase
schedule are external inputs; all digitization, carry logic, and accumulation
are model operations. Feedback is copied unchanged between requests.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

import numpy as np

WIDTH, WORK, HEAD, KV, NH = 1280, 640, 256, 4, 8
ONE, ZERO, NORMAL, NEGATIVE, EXTERNAL_ONE = range(5)
ACC, Q, X, G, UC, UF, US, A = 16, 48, 64, 96, 114, 115, 116, 128
SEED_REF, SCALE, WRITE_ADDR, READ_ADDR, GATES, PAYLOAD = 148, 149, 168, 184, 200, 264
ACT, PLANE, REF, SCALE_OUT = 1200, 1210, 1211, 1220
SEED, COARSE, FINE, SUM = range(4)
NBITS, GAIN, UPGAIN = 18, 1048576, 128
INGRESS = (*range(5), *range(ACC, ACC+NBITS), *range(Q, Q+6),
           *range(X, X+NBITS), *range(G, G+NBITS), UC, UF, US,
           *range(A, A+NBITS), SEED_REF, *range(SCALE, SCALE+16),
           *range(WRITE_ADDR, WRITE_ADDR+16), *range(READ_ADDR, READ_ADDR+16),
           *range(GATES, GATES+64), *range(PAYLOAD, PAYLOAD+256))


class FF:
    def __init__(self):
        self.g, self.u, self.d = [], [], []

    def row(self, gate, up, down):
        self.g.append(gate)
        self.u.append(up)
        self.d.append(down)

    def const(self, up, down):
        for sign in (1, -1):
            self.row({WORK+ONE: 4*sign}, {up: UPGAIN},
                     {dst: coef*sign for dst, coef in down.items()})

    def step(self, terms, up, down, gain=512):
        # SiLU(t+4u)-SiLU(t-4u)-SiLU(4u)+SiLU(-4u)
        # approaches 4u*sign(t), where u is the common neural ONE.
        for include, bias, sign in ((1,4,1),(1,-4,-1),(0,4,-1),(0,-4,1)):
            gate = {lane: coef*gain for lane, coef in terms.items()} if include else {}
            gate[WORK+ONE] = gate.get(WORK+ONE, 0) + bias
            self.row(gate, {up: UPGAIN}, {dst: coef*sign for dst, coef in down.items()})

    def buf(self, src, up, down):
        self.step({src: 1}, up, down)

    def match(self, terms, up, down):
        predicate = dict(terms)
        predicate[WORK+ONE] = predicate.get(WORK+ONE, 0) + 1-len(terms)
        self.step(predicate, up, down)

    def xor(self, a, b, bsign, up, down):
        terms = {a: 1, b: bsign}
        for bias, polarity in ((1,1),(-1,-1)):
            self.step(terms | {WORK+ONE: bias}, up,
                      {dst: coef*polarity for dst,coef in down.items()})
        self.const(up, {dst: -coef for dst,coef in down.items()})

    def matrices(self):
        n = (len(self.g)+255)//256*256
        gate, up, down = np.zeros((n,WIDTH),np.float32), np.zeros((n,WIDTH),np.float32), np.zeros((WIDTH,n),np.float32)
        for row, (gs,us,ds) in enumerate(zip(self.g,self.u,self.d)):
            for col,coef in gs.items(): gate[row,col] += coef
            for col,coef in us.items(): up[row,col] += coef
            for col,coef in ds.items(): down[col,row] += coef
        return gate, up, down


def compile_ff(restart=True, canonical_a0=False):
    f = FF()
    # A cyclic neural phase unit gates each stage, including its own successor.
    for current, target in ((UC,UF),(UF,US),(US,UC)):
        down={ONE:1,ZERO:-1}
        if restart or current!=US: down[target]=1
        f.const(WORK+current, down)
    for bit in range(NBITS):
        f.buf(WORK+ACC+bit, WORK+UC, {ACC+bit:1})
        if bit==0 and canonical_a0: f.const(WORK+UC,{A+bit:-1})
        else: f.buf(WORK+(ZERO if bit == 0 else ACC+bit-1), WORK+UC, {A+bit:1})
    # The negated signed activation plane can reach +512, so q needs six
    # bits and 32 thresholds (rather than the unsigned-plane 30 thresholds).
    for j in range(1,33):
        down = {}
        for bit in range(6):
            if j % (1<<bit) == 0:
                down[Q+bit] = 1 if (j//(1<<bit))%2 else -1
        f.step({PLANE:1, REF:-(32*j-512-.5)/512}, WORK+UC, down, GAIN)
    # Bits with an even number of transitions need the remaining constant.
    for bit in range(6):
        n = 32//(1<<bit)
        c = -1 + (n%2)
        if c: f.const(WORK+UC,{Q+bit:c})
    # Fine r=(plane+512)-32q. Build low operand bits and XOR/generate in
    # the same FFN by using the neural A bit as the SiLU up operand.
    for bit in range(5):
        f.const(WORK+A+bit,{G+bit:.5})
        f.const(WORK+UF,{G+bit:-.5})
        for j in range(1,32):
            if j % (1<<bit): continue
            polarity = 1 if (j//(1<<bit))%2 else -1
            terms = {PLANE:1, REF:-(j-.5)/512}
            f.step(terms,WORK+UF,{G+bit:.5*polarity},GAIN)
            f.step(terms,WORK+A+bit,{X+bit:-polarity,G+bit:.5*polarity},GAIN)
    # The signed plane is U xor 512, so bit9 is the inverted high q bit.
    for bit in range(5,10):
        qbit, polarity = bit-5, (1 if bit < 9 else -1)
        a, b = WORK+A+bit, WORK+Q+qbit
        f.xor(a,b,polarity,WORK+UF,{X+bit:1})
        f.match(((a,1),(b,polarity)),WORK+UF,{G+bit:1})
    # P=U-512 has a positive sign only when both U9 and U10 are zero.
    for bit in range(10,NBITS):
        a,q4,q5=WORK+A+bit,WORK+Q+4,WORK+Q+5
        n=0
        for av in (-1,1):
            for v4 in (-1,1):
                for v5 in (-1,1):
                    b=1 if v4<0 and v5<0 else -1
                    if av!=b:
                        f.match(((a,av),(q4,v4),(q5,v5)),WORK+UF,{X+bit:1}); n+=1
        f.const(WORK+UF,{X+bit:n-1})
        f.match(((a,1),(q4,-1),(q5,-1)),WORK+UF,{G+bit:1})
    # Carry terms are mutually exclusive. C_i=sum(T_j)+(n-1).
    # XOR with X_i is multiplication by -X_i, supplied by the up matrix.
    f.buf(WORK+X,WORK+US,{ACC:1})
    for bit in range(1,NBITS):
        for j in range(bit):
            terms = ((WORK+G+j,1), *((WORK+X+k,1) for k in range(j+1,bit)))
            f.match(terms,WORK+X+bit,{ACC+bit:-1})
        if bit > 1: f.const(WORK+X+bit,{ACC+bit:-(bit-1)})
    return f


def map_ff_rows(ports, work_offset, plane_source, reference_source, restart=False):
    """Return composable F32 sparse GGUF coefficients, with row indices from zero.

    ports maps ONE/ZERO/UC/UF/US to scalar raw-output lane IDs, and maps
    ACC/A/X/G to 18 raw lanes each, Q to six raw lanes. Source feedback is
    projected by the containing model to work_offset + each raw lane.
    plane_source/reference_source are attention-output destinations, already
    inside the layer; plane_source is the weighted plane divided by 512.

    ONE/ZERO/A/X/UC/UF/US must be exclusive ephemeral outputs: omit them
    from Circuit.persistent and do not write them in unrelated operations.
    Suppress ordinary Circuit output rows at every destination written by
    these rows during the three dot stages (especially SUM accumulator).
    With restart=False, the final VA read must initialize ONE/ZERO/UC.
    """
    layout={'ONE':(ONE,1),'ZERO':(ZERO,1),'ACC':(ACC,NBITS),'Q':(Q,6),
            'X':(X,NBITS),'G':(G,NBITS),'UC':(UC,1),'UF':(UF,1),'US':(US,1),'A':(A,NBITS)}
    mapping={}
    for name,(base,count) in layout.items():
        actual=(ports[name],) if count==1 else tuple(ports[name])
        if len(actual)!=count: raise ValueError(f'{name}: expected {count} ports')
        mapping.update(zip(range(base,base+count),actual))
    if len(set(mapping.values()))!=len(mapping): raise ValueError('Overlapping dot ports')
    def source(lane):
        if lane==PLANE: return plane_source
        if lane==REF: return reference_source
        return work_offset+mapping[lane-WORK]
    circuit=compile_ff(restart)
    result={'gate':[],'up':[],'down':[],'rows':len(circuit.g)}
    for row,(g,u,d) in enumerate(zip(circuit.g,circuit.u,circuit.d)):
        result['gate'].extend(((row,source(col)),coef) for col,coef in g.items() if coef)
        result['up'].extend(((row,source(col)),coef) for col,coef in u.items() if coef)
        result['down'].extend(((mapping[dst],row),coef) for dst,coef in d.items() if coef)
    return result


def init_ff_rows(ports, global_one_source, phase_action_source):
    """Two rows in the last VA-read phase initialize the ephemeral dot clock.

    The up source is the containing model's one-hot last-VA-read action lane,
    so these rows emit zero during every other phase. Accumulator reset to
    zero is a separate normal Circuit effect at the start of the chunk.
    """
    result={'gate':[],'up':[],'down':[],'rows':2}
    for row,sign in enumerate((1,-1)):
        result['gate'].append(((row,global_one_source),4*sign))
        result['up'].append(((row,phase_action_source),UPGAIN))
        for name,polarity in (('ONE',1),('ZERO',-1),('UC',1)):
            result['down'].append(((ports[name],row),sign*polarity))
    return result


def action_ff_rows(ports, work_offset, plane_source, reference_source, phase_actions,
                   neural_up_gain=1024, boolean_gain=65536):
    """Composition candidate: active action sets each SiLU plateau bias.

    This keeps the neural multiplication but removes the quadratic local-ONE
    amplitude loss next to a large background of standard Circuit refreshes.
    phase_actions maps COARSE/FINE/SUM to tuples of absolute action lanes.
    COARSE uses standard up16*action, FINE/SUM use neural up1024*A/X.
    reference_source must be (REF, REF2), two exact copies of row v. Coarse
    half-integer thresholds split their coefficient into a BF16-exact integer
    multiple of 65536 on REF and +1024 on REF2. Every FFN coefficient is then
    exactly BF16-representable. Local ONE sets Boolean predicate offsets,
    the active action used for the centered difference's four-unit shift.
    A/X and UF/US must be exclusive ephemeral ports. No cyclic restart or
    initialization rows are needed; the model action selects COARSE.
    """
    layout={'ONE':(ONE,1),'ZERO':(ZERO,1),'ACC':(ACC,NBITS),'Q':(Q,6),
            'X':(X,NBITS),'G':(G,NBITS),'UC':(UC,1),'UF':(UF,1),'US':(US,1),'A':(A,NBITS)}
    mapping={}
    for name,(base,count) in layout.items():
        actual=(ports[name],) if count==1 else tuple(ports[name])
        if len(actual)!=count: raise ValueError(f'{name}: expected {count} ports')
        mapping.update(zip(range(base,base+count),actual))
    if len(set(mapping.values()))!=len(mapping): raise ValueError('Overlapping dot ports')
    refs=tuple(reference_source)
    if len(refs)!=2: raise ValueError('Two reference copies are required for BF16 coarse thresholds')
    def source(lane):
        if lane==PLANE: return plane_source
        if lane==REF: return refs[0]
        return work_offset+mapping[lane-WORK]
    circuit=compile_ff(False,True); result={'gate':[],'up':[],'down':[],'rows':len(circuit.g)}
    for row,(g,u,d) in enumerate(zip(circuit.g,circuit.u,circuit.d)):
        upsrc=next(iter(u)); local=upsrc-WORK
        phase='COARSE' if local==UC else 'FINE' if local==UF or A<=local<A+NBITS else 'SUM'
        actions=tuple(phase_actions[phase]); analog=PLANE in g or REF in g
        combined=g.get(WORK+ONE,0)
        bias=combined-round(combined/512)*512
        if abs(bias)!=4: raise ValueError(f'Unexpected plateau bias {bias}')
        predicate=combined-bias
        scale=1 if analog else boolean_gain/512
        for col,coef in g.items():
            if col!=WORK+ONE and coef:
                if col==REF and phase=='COARSE':
                    if coef-1024: result['gate'].append(((row,refs[0]),coef-1024))
                    result['gate'].append(((row,refs[1]),1024))
                else: result['gate'].append(((row,source(col)),coef*scale))
        if predicate: result['gate'].append(((row,source(WORK+ONE)),predicate*scale))
        result['gate'].extend(((row,action),bias) for action in actions)
        if phase=='COARSE': result['up'].extend(((row,action),16) for action in actions)
        else: result['up'].append(((row,source(upsrc)),neural_up_gain))
        result['down'].extend(((mapping[dest],row),coef) for dest,coef in d.items() if coef)
    return result


def weights():
    f = compile_ff()
    g,u,d = f.matrices()
    q,k,v,o = (np.zeros(s,np.float32) for s in
               ((2*HEAD*NH,WIDTH),(HEAD*KV,WIDTH),(HEAD*KV,WIDTH),(WIDTH,HEAD*NH)))
    for group in range(KV):
        for sub in range(2):
            head = group*2+sub
            start = 2*HEAD*head
            for bit in range(16):
                for rep in range(8):
                    q[start+64+8*bit+rep,WORK+READ_ADDR+bit] = 1
            for lane in range(16):
                for bit in range(4):
                    value_lane = bit*16+lane
                    gate_row = start+HEAD+value_lane
                    q[gate_row,WORK+GATES+16*group+lane] = 65536
                    q[gate_row,WORK+(NORMAL if sub==0 else NEGATIVE)] = 65536
                    q[gate_row,WORK+EXTERNAL_ONE] = -65536
                    q[gate_row,ACT+SEED] = -65536
                    q[gate_row,ACT+SUM] = -65536
                    o[PLANE,HEAD*head+value_lane] = (1 if sub==0 else -1)*(1,2,4,-8)[bit]/512
        for bit in range(16):
            for rep in range(8): k[HEAD*group+64+8*bit+rep,WORK+WRITE_ADDR+bit] = 1
        for bit in range(4):
            for lane in range(16):
                v[HEAD*group+bit*16+lane,WORK+PAYLOAD+64*group+16*bit+lane] = 1
    # The first query head also returns row reference, q correction, and scale16.
    for lane in range(64,88):
        v[lane,WORK+SEED_REF if lane<72 else WORK+SCALE+lane-72] = 1
        q[HEAD+lane,ACT+COARSE] = 65536
        q[HEAD+lane,ACT+FINE] = 65536
        q[HEAD+lane,ACT+SEED] = -65536
        q[HEAD+lane,ACT+SUM] = -65536
    o[REF,64] = 1
    for bit in range(6):
        lane=65+bit
        q[HEAD+lane,:] = 0
        q[HEAD+lane,WORK+Q+bit] = 65536
        for token in (SEED,COARSE,SUM): q[HEAD+lane,ACT+token] = -65536
        o[PLANE,lane] = -(32<<bit)/512
    q[HEAD+71,ACT+COARSE] = -65536
    o[PLANE,71] = 1
    for bit in range(16):
        o[SCALE_OUT+bit,72+bit] = 2
        o[SCALE_OUT+bit,64] = -1
    return f,(g,u,d),(q,k,v,o)


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def export(path):
    from gguf import GGUFWriter
    if path.exists(): raise FileExistsError(path)
    if shutil.disk_usage(path.parent).free < 1024**3: raise RuntimeError('Need 1 GiB free before small export')
    f,(g,u,d),(q,k,v,o) = weights()
    w = GGUFWriter(path,'qwen35')
    for name,value in {'context_length':256,'embedding_length':WIDTH,'vocab_size':16,
        'feed_forward_length':len(g),'attention.head_count':NH,'attention.head_count_kv':KV,
        'attention.key_length':HEAD,'attention.value_length':HEAD,'rope.dimension_count':64,
        'ssm.conv_kernel':4,'ssm.state_size':HEAD,'ssm.group_count':1,'ssm.time_step_rank':1,
        'ssm.inner_size':HEAD,'full_attention_interval':4}.items(): w.add_uint32('qwen35.'+name,value)
    w.add_array('qwen35.rope.dimension_sections',[16,8,8,0])
    w.add_float32('qwen35.attention.layer_norm_rms_epsilon',1e-6)
    w.add_float32('qwen35.rope.freq_base',10000.)
    w.add_string('tokenizer.ggml.model','none')
    w.add_name('Isolated stock 64-lane gate-plane dot')
    w.add_block_count(2)
    w.add_uint32('qwen35.nextn_predict_layers',1)
    w.add_tensor('token_embd.weight',np.zeros((16,WIDTH),np.float32))
    w.add_tensor('output_norm.weight',np.ones(WIDTH,np.float32))
    def add(name,data): w.add_tensor('blk.1.'+name,data)
    unit=np.float32(1/math.sqrt(WIDTH))
    emb=np.zeros((16,WIDTH),np.float32)
    for token in range(4): emb[token,ACT+token]=1
    add('nextn.embed_tokens.weight',emb)
    add('nextn.enorm.weight',np.full(WIDTH,unit,np.float32))
    hn=np.zeros(WIDTH,np.float32); hn[list(INGRESS)]=unit
    add('nextn.hnorm.weight',hn)
    proj=np.zeros((WIDTH,2*WIDTH),np.float32)
    for token in range(4): proj[ACT+token,ACT+token]=1
    for lane in INGRESS: proj[WORK+lane,WIDTH+lane]=1
    add('nextn.eh_proj.weight',proj)
    add('attn_norm.weight',np.full(WIDTH,unit,np.float32))
    for name,data in zip(('attn_q','attn_k','attn_v','attn_output'),(q,k,v,o)): add(name+'.weight',data)
    kn=np.zeros(HEAD,np.float32); kn[64:192]=8
    add('attn_q_norm.weight',kn); add('attn_k_norm.weight',kn)
    add('post_attention_norm.weight',np.full(WIDTH,unit,np.float32))
    for name,data in zip(('ffn_gate','ffn_up','ffn_down'),(g,u,d)): add(name+'.weight',data)
    add('nextn.shared_head_norm.weight',np.ones(WIDTH,np.float32))
    scores=np.zeros((16,WIDTH),np.float32)
    for bit in range(8): scores[2*bit,ACC+bit]=1; scores[2*bit+1,ACC+bit]=-1
    add('nextn.shared_head_head.weight',scores)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    report=dict(width=WIDTH,query_heads=NH,kv_heads=KV,ffn_rows_used=len(f.g),ffn_rows=len(g),
        all_matrices='F32',kv_cache='F16',decodes_per_dot=24,bytes=path.stat().st_size,sha256=digest(path))
    path.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def fixtures(path):
    rng=np.random.default_rng(913742)
    rows=[]
    patterns=[np.full(64,-8),np.full(64,7),np.zeros(64),np.resize(np.arange(-8,8),64)]
    patterns += [rng.integers(-8,8,64) for _ in range(12)]
    for i,ww in enumerate(patterns): rows.append((i, int(rng.integers(0,65536)), ww.astype(int).tolist()))
    cases=[]
    for row,_,ww in rows:
        aa=[np.full(64,-128),np.full(64,127),np.zeros(64),np.resize([-128,127],64)]
        aa += [rng.integers(-128,128,64) for _ in range(6)]
        for a in aa: cases.append((row,a.astype(int).tolist(),int(np.dot(a,np.array(ww)))))
    # Adjacent addresses have conflicting rows; exact threshold ties and one-hot lanes.
    for lane in range(64):
        a=np.zeros(64,dtype=int); a[lane]=(-128,127,-1,1)[lane%4]
        row=lane%len(rows); cases.append((row,a.tolist(),int(np.dot(a,rows[row][2]))))
    cases += cases[:8]
    with path.open('wb') as stream:
        stream.write(struct.pack('<II',len(rows),len(cases)))
        for addr,scale,ww in rows: stream.write(struct.pack('<II64B',addr,scale,*[w+8 for w in ww]))
        for addr,aa,dot in cases: stream.write(struct.pack('<I64bi',addr,*aa,dot))
    path.with_suffix('.json').write_text(json.dumps(dict(rows=rows,cases=cases),indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),cases=len(cases),sha256=digest(path))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('mode',choices=('plan','export','fixtures')); p.add_argument('path',nargs='?',type=Path)
    args=p.parse_args()
    if args.mode=='plan':
        f=compile_ff(); print(json.dumps(dict(width=WIDTH,ffn_rows_used=len(f.g),ffn_rows=(len(f.g)+255)//256*256,decodes_per_dot=24),indent=2))
    elif args.mode=='export': export(args.path)
    else: fixtures(args.path)
