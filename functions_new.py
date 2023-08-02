from collections import OrderedDict
from evcouplings.align import Alignment, map_matrix,read_fasta

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def msa_protocol(name_msa_file):
  ### name_msa_file: path to msa fime of gene
  ### return: alignment of gene

  with open(name_msa_file, "r") as infile:
    #seqs = OrderedDict()
    next(infile)

    #for i, (seq_id, seq) in enumerate(read_fasta(infile)):
    proper_infile = read_a3m(infile, inserts = "delete") # convert from a3m to a2m
    #seq = seq.replace(".", "")
    # seqs[seq_id] = seq
    #n_items = take(n, seqs.items())

    #aln = Alignment.from_file(proper_infile, format="fasta")
    aln = Alignment.from_dict(proper_infile)

  # Sequence length and number of sequences
  print(f"alignment is of length {aln.L} and has {aln.N} sequences")

  # Protocol Hopf
  # calculate the percent identity of every sequence in the alignment to the first sequence
  ident_perc = aln.identities_to(aln.matrix[0])
  ident_perc_list = ident_perc.tolist()

  # keep identifiers with > 50 percentage identity and colunns with at least 70% occupancy
  index_keep = []
  for i, iden in enumerate(ident_perc_list):
    if iden > 0.5: # 0.5= sequences with at least 50% of identity to the frst sequence are kept
      index_keep.append(i)

  #use the "count" method of the class  -  Percentage of gaps
  maximum1 = aln.count(axis="seq",char="-")#.argmax()

  filtered_ind = [i for i in range(len(maximum1)) if maximum1[i] <= 0.3] # 0.3 30% of gaps
  sequences_to_keep = intersection(index_keep, filtered_ind) # keep indeces that satisfy both conditions

  selection_index = sequences_to_keep
  aln_subsection = aln.select(sequences=selection_index)
  print(f"the new alignment has {aln_subsection.N} sequences")

  # if remaining sequences in MSA < 15 redo the process with less strict filtering
  if aln_subsection.N <15:
    index_keep = []
    for i, iden in enumerate(ident_perc_list):
      #if iden > 0.05: # 0.3= sequences with at least 5% of identity to the frst sequence are kept
      if iden > 0.27: # 0.3= sequences with at least 10% of identity to the frst sequence are kept
        index_keep.append(i)
    filtered_ind = [i for i in range(len(maximum1)) if maximum1[i] <= 0.7] # max 60% of gaps
    sequences_to_keep = intersection(index_keep, filtered_ind) # keep indeces that satisfy both conditions
    selection_index = sequences_to_keep
    aln_subsection = aln.select(sequences=selection_index)

  if aln_subsection.N <15:
    index_keep = []
    for i, iden in enumerate(ident_perc_list):
      #if iden > 0.05: # 0.3= sequences with at least 5% of identity to the frst sequence are kept
      if iden > 0.2: # 0.3= sequences with at least 20% of identity to the frst sequence are kept
        index_keep.append(i)
    filtered_ind = [i for i in range(len(maximum1)) if maximum1[i] <= 0.7] # max 60% of gaps
    sequences_to_keep = intersection(index_keep, filtered_ind) # keep indeces that satisfy both conditions
    selection_index = sequences_to_keep
    aln_subsection = aln.select(sequences=selection_index)
  #if aln_subsection.N >5000:
  #  index_keep = []
  #  for i, iden in enumerate(ident_perc_list):
  #    if iden > 0.6: # 0.4= sequences with at least 30% of identity to the frst sequence are kept
  #      index_keep.append(i)
  #  filtered_ind = [i for i in range(len(maximum1)) if maximum1[i] <= 0.2] # 0.6 60% of gaps
  #  sequences_to_keep = intersection(index_keep, filtered_ind) # keep indeces that satisfy both conditions
  #  selection_index = sequences_to_keep
  #  aln_subsection = aln.select(sequences=selection_index)


  #use the "count" method of the class  -  Percentage of gaps
  #maximum1 = aln.count(axis="seq",char="-")#.argmax()
  #filtered_ind = [i for i in range(len(maximum1)) if maximum1[i] <= 0.3] # 0.6 60% of gaps
  print(f"the new alignment has {aln_subsection.N} sequences")

  return aln_subsection

def read_a3m(fileobj, inserts="first"):
    """
    Read an alignment in compressed a3m format and expand
    into a2m format.
    .. note::
        this function is currently not able to keep inserts in all the sequences
    ..todo::
        implement this
    Parameters
    ----------
    fileobj : file-like object
        A3M alignment file
    inserts : {"first", "delete"}
        Keep inserts in first sequence, or delete
        any insert column and keep only match state
        columns.
    Returns
    -------
    OrderedDict
        Sequences in alignment (key: ID, value: sequence),
        in order they appeared in input file
    Raises
    ------
    ValueError
        Upon invalid choice of insert strategy
    """
    seqs = OrderedDict()

    for i, (seq_id, seq) in enumerate(read_fasta(fileobj)):
        # remove any insert gaps that may still be in alignment
        # (just to be sure)
        seq = seq.replace(".", "")

        if inserts == "first":
            # define "spacing" of uppercase columns in
            # final alignment based on target sequence;
            # remaining columns will be filled with insert
            # gaps in the other sequences
            if i == 0:
                uppercase_cols = [
                    j for (j, c) in enumerate(seq)
                    if (c == c.upper() or c == "-")
                ]
                gap_template = np.array(["."] * len(seq))
                filled_seq = seq
            else:
                uppercase_chars = [
                    c for c in seq if c == c.upper() or c == "-"
                ]
                filled = np.copy(gap_template)
                filled[uppercase_cols] = uppercase_chars
                filled_seq = "".join(filled)

        elif inserts == "delete":
            # remove all lowercase letters and insert gaps .;
            # since each sequence must have same number of
            # uppercase letters or match gaps -, this gives
            # the final sequence in alignment
            seq = "".join([c for c in seq if c == c.upper() and c != "."])
        else:
            raise ValueError(
                "Invalid option for inserts: {}".format(inserts)
            )

        seqs[seq_id] = seq

    return seqs
