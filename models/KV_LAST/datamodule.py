import torch
from utils.vocab import vocab
from .line_pos_enc import InLinePos
from utils.make_mask import glue_line_triu

class Task_batchs():
    def __init__(self,task_name, bz, max_len):
        super(Task_batchs, self).__init__()
        self.task_name=task_name
        self.bz=0
        self.gts=[]
        self.input=torch.zeros((bz, max_len), dtype=torch.long)
        self.tgt=torch.zeros((bz, max_len), dtype=torch.long)
        self.pos=torch.zeros((bz, max_len, 256))
        self.li=torch.zeros((bz, max_len), dtype=torch.long)
        self.attn_mask=torch.ones((bz, max_len, max_len), dtype=torch.bool)
        self.dest=torch.full((bz, max_len), fill_value=-1, dtype=torch.long)
        self.lens=[]
    def to(self, device):
        self.input=self.input.to(device)
        self.tgt=self.tgt.to(device)
        self.attn_mask=self.attn_mask.to(device)
        self.li=self.li.to(device)
        self.pos=self.pos.to(device)
        return self
    
class BiMultinline():
    def __init__(self, task_name, device, d_model, nline):
        super(BiMultinline, self).__init__()
        self.nlines = nline
        self.device = device
        self.d_model = d_model
        self.task_name = task_name

        self.lines = [[vocab.word2idx[f'6:sos_{i}']] for i in range(self.nlines)]
        self.lines_r = [[vocab.word2idx[f'6:eos_{i}']] for i in range(self.nlines)]
        self.line_lens =[2]*self.nlines

        self.tmp_line_lens = []
        self.tmp = []

        self.done = [False]*self.nlines
        self.poser = InLinePos(256)
        self.max_len=256

    def make_input(self):
        pt=0
        false_count = self.done.count(False)
        self.batch=Task_batchs(task_name=self.task_name, bz=1, max_len=2*false_count).to(self.device)
        for i in range(self.nlines):
            this_len=2
            if not self.done[i]:
                if pt+this_len>self.max_len-1: 
                    self.done[i]=True
                    return
                self.batch.input[0][pt:pt+this_len]=torch.tensor([self.lines[i][-1], self.lines_r[i][0]])
                self.tmp_line_lens.append(self.line_lens[i])
                self.tmp.append(i)
                self.batch.li[0][pt:pt+this_len]=i
                pt+=this_len
        # self.batch.attn_mask[0]=glue_line_triu(self.line_lens, bidir=True, vis_other=True, padding=sum(self.line_lens))
        self.batch.pos[0] =self.poser.get_a_poser(self.tmp_line_lens, task_name=self.batch.task_name)
    
    def update(self, new_char_outputs): # Such dynamic list conversion is prototype. Consider using a pre defined two dim dual-end structure in deployment.
        out_chars = new_char_outputs[0].argmax(-1)
        # print(vocab.lindices2llabel(self.lines))
        # print("===========================================")
        # print(vocab.lindices2llabel(self.lines_r))
        # print(self.line_lens)
        # print("-------------------------------------------")
        # print(self.line_lens)

        index = [2 * i for i,_ in enumerate(self.tmp_line_lens)]
        index_r = [i+1 for i in index]

        these_chars = out_chars[index]
        these_chars_r = out_chars[index_r]

        for i in range(len(self.tmp_line_lens)):
            if self.done[self.tmp[i]]: continue

            this_char = these_chars[i]
            this_char_r = these_chars_r[i]

            if 'mol' in vocab.idx2word[this_char.item()] and 'mor' in vocab.idx2word[this_char_r.item()]:
                self.done[self.tmp[i]]=True
                continue

            self.lines[self.tmp[i]].append(this_char.item())
            self.lines_r[self.tmp[i]]=[this_char_r.item()]+self.lines_r[self.tmp[i]]
            self.line_lens[self.tmp[i]]+=2
        self.tmp = []
        self.tmp_line_lens = []


    def is_done(self):
        if sum(self.done)==self.nlines: return True
        if sum([len(line) for line in self.lines])>=(self.max_len-self.nlines*2)//2: return True
        return  False

    def return_ans(self):
        ans=[]
        for i in range(self.nlines):
            this_line=[]
            for char in self.lines[i]+self.lines_r[i]:
                if 'sos' not in vocab.idx2word[char] and 'eos' not in vocab.idx2word[char] and 'mor' not in vocab.idx2word[char] and 'mol' not in vocab.idx2word[char]:
                    this_line.append(char)
            ans.append(this_line)
        return ans

    def print(self):
        for i in range(self.nlines):
            print(self.lines[i],' + ',self.lines_r[i])

class Plainline():
    def __init__(self, task_name, device, d_model):
        super(Plainline, self).__init__()

        self.device = device
        self.d_model = d_model
        if task_name == 'plain_l2r': starter=vocab.word2idx['0:sos']
        if task_name == 'plain_r2l': starter=vocab.word2idx['1:sos']
        self.task_name = task_name

        self.poser=InLinePos(256)
        self.lines = [starter]
        self.done = False
        

    def make_input(self):
        this_len=1
        lines_len = len(self.lines)
        self.batch=Task_batchs(task_name=self.task_name, bz=1, max_len=this_len).to(self.device)
        if len(self.lines)>255: return self.batch
        self.batch.input[0][:this_len]=torch.tensor(self.lines[-1]).to(self.device)
        # self.batch.attn_mask[0]=glue_line_triu([this_len],padding=256).to(self.device)
        self.batch.pos[0,:sum([this_len])]=self.poser.get_a_poser([lines_len],task_name=self.batch.task_name)


    def update(self, new_char_outputs):
        
        out_chars=new_char_outputs[0].argmax(-1)
        this_char=out_chars
        if this_char.item()!=vocab.word2idx['0:eos'] and this_char.item()!=vocab.word2idx['1:eos']:
            self.lines.append(this_char.item())
        else:
            self.done=True
            
    def is_done(self):
        if self.done: return True
        if len(self.lines)>=256-2: return True
        return  False

    def return_ans(self):
        ans=[]
        for char in self.lines[1:]:
            if 'sos' not in vocab.idx2word[char] and 'eos' not in vocab.idx2word[char]:
                ans.append(char)
        return ans

    def print(self):
        print(self.lines)