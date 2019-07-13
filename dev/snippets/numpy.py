def _generate_einstr(ndim, a_axes, b_axes):
    # Set up Einstein index strings for operand a and result
    a_einstr = np.array(list(letter_range(ndim)))
    broadcast_str = a_einstr[[i not in a_axes for i in range(ndim)]]
    # Set up Einstein index string for operand b
    idx = np.argsort(b_axes)
    b_axes = np.array(b_axes)[idx]
    contracted_str = a_einstr[np.array(a_axes)][idx]
    b_einstr = list(broadcast_str)
    for i, cs in enumerate(contracted_str):
        b_einstr.insert(b_axes[i], cs)
    # Assemble Einstein index string
    einstr = ("".join(a_einstr) + "," + "".join(b_einstr)
              + "->" + "".join(broadcast_str))
    return einstr
