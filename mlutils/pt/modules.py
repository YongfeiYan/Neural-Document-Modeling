

# embedding creation

# extract last output in last time  step


def extract_last_timestep(output, lengths, batch_first):
    """Get the output of last time step.
    output: seq_len x batch_size x dim if not batch_first. Else batch_size x seq_len x dim
    length: one dimensional torch.LongTensor of lengths in a batch.
    """
    idx = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    idx = idx.unsqueeze(time_dimension)
    if output.is_cuda:
        idx = idx.cuda(output.data.get_device())
    return output.gather(time_dimension, idx).squeeze(time_dimension)

