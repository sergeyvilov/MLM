#!/usr/bin/env python
# coding: utf-8

# In[3]:


data_dir = '/s/project/mll/sergey/effect_prediction/MLM/'


# In[4]:


train_fasta = data_dir + 'fasta/240_mammals/240_mammals.shuffled.fa'
test_fasta = data_dir + 'fasta/240_mammals/species/Homo_sapiens.fa'

output_dir = data_dir + 'dnabert_3utr/data/'


# In[ ]:


print(f'output dir: {output_dir}')


# In[16]:


def kmers_stride1(seq, k=6):
    # splits a sequence into overlapping k-mers
    return [seq[i:i + k] for i in range(0, len(seq)-k+1)] 

def chunkstring(string, length):
    # chunks a string into segments of length
    return (string[0+i:length+i] for i in range(0, len(string), length))


# In[22]:


def dump_seq(seq, f_out):
    for seq_chunk in chunkstring(seq, 510):
        if len(seq_chunk)<5:
            break
        k_mers = kmers_stride1(seq_chunk)
        seq_chunk = ' '.join(k_mers)
        f_out.write(seq_chunk + '\n')


def convert_sequences(input_fasta, dnabert_txt):
    
    seq = ''
    
    c = 0
    
    with open(input_fasta, 'r') as f_in:
        with open(dnabert_txt, 'w') as f_out:
            for line in f_in:
                if line.startswith('>'):
                    dump_seq(seq,f_out)
                    seq = ''
                    c+=1
                    if c%10000==0:
                        print(f'{c} sequences processed')
                else: 
                    seq += line.rstrip().upper()
            dump_seq(seq,f_out)


# In[ ]:


print(f'converting test FASTA: {test_fasta}')

convert_sequences(test_fasta,output_dir + 'Homo_sapiens_6kmer.txt')

print(f'converting train FASTA: {train_fasta}')

convert_sequences(train_fasta,output_dir + '240_species_shuffled_6kmer.txt')

