"""A three-phase exact small adder with six-bit local carry predicates."""


def emit_add(c, *, entry, finish, left, right, destination, partial, carry0, carry1,
             block_carries, state, logic, chunk=6, carry_in=None):
    """Add equal-width signed-Boolean words modulo their width.

    ``left/right`` are read lanes; all other tuples are caller-owned physical
    scratch.  At most three chunks are supported.  Gate coefficients are at
    most 32 for the default chunk size, avoiding a long weighted comparator.
    """
    width = len(left)
    if len(right) != width or any(len(x) != width for x in (destination, partial, carry0, carry1)):
        raise ValueError('Adder words must have the same width')
    blocks = (width + chunk - 1) // chunk
    if not 1 <= blocks <= 3 or len(block_carries) < blocks - 1:
        raise ValueError('Adder requires one to three chunks and their carry lanes')
    middle, output = entry + '_CARRIES', entry + '_SUM'
    zero = logic.WORK + logic.ZERO
    incoming0 = zero if carry_in is None else carry_in
    for bit, (a, b) in enumerate(zip(left, right)):
        start = bit // chunk * chunk
        terms = tuple((left[j], 1 << (j - start)) for j in range(start, bit + 1)) + tuple(
            (right[j], 1 << (j - start)) for j in range(start, bit + 1))
        c.put(entry, partial[bit], logic.xor(a, b))
        if bit < chunk:
            c.put(entry, carry0[bit], logic.threshold((*terms, (incoming0, 1))))
        else:
            c.put(entry, carry0[bit], logic.threshold(terms, -1))
            c.put(entry, carry1[bit], logic.threshold(terms, 1))
        if bit == chunk - 1 and blocks > 1:
            c.put(entry, block_carries[0], logic.threshold((*terms, (incoming0, 1))))
    if blocks > 2:
        c.route(entry, middle)
        end = 2 * chunk - 1
        c.put(middle, block_carries[1], logic.threshold(((state(carry0)[end], 2),
            (state(carry1)[end], 1), (state(block_carries)[0], 1)), 1))
        c.route(middle, output)
    else:
        c.route(entry, output)
    for bit, dest in enumerate(destination):
        block, local = divmod(bit, chunk)
        incoming = incoming0 if block == 0 else state(block_carries)[block - 1]
        x = state(partial)[bit]
        if local == 0:
            c.put(output, dest, logic.xor(x, incoming))
        elif block == 0:
            c.put(output, dest, logic.xor(x, state(carry0)[bit - 1]))
        else:
            g0, g1 = state(carry0)[bit - 1], state(carry1)[bit - 1]
            cases = (((x, 1), (g1, -1)),
                     ((x, 1), (g1, 1), (g0, -1), (incoming, -1)),
                     ((x, -1), (g0, 1)),
                     ((x, -1), (g0, -1), (g1, 1), (incoming, 1)))
            c.put(output, dest, (*(term for case in cases for term in logic.match(case)),
                                 *logic.constant(3)))
    c.route(output, finish)
