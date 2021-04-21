import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.activations import swish,gelu_new
from config import Config
from torch.autograd import Variable
import math
from pytorch_transformers.tokenization_bert import BertTokenizer
torch.set_printoptions(precision=None, threshold=1000000, edgeitems=10, linewidth=100000, profile=None)

ACT2FN = {"gelu": gelu_new, "relu":F.relu, "swish":swish}
#TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
class VDSelfAttention(nn.Module):

    def __init__(self,config):
        super(VDSelfAttention,self).__init__()
        self.config = config

        if config.d_model % config.n_head != 0 :
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention"
                "heads (%d)" % (config.d_model, config.n_head)
            )

        self.output_attentions = config.output_attentions

        self.n_head = config.n_head
        self.d_head = int(config.d_model / config.n_head)
        self.all_head_size = self.n_head * self.d_head

        self.query = nn.Linear(config.d_model, self.all_head_size)
        self.key = nn.Linear(config.d_model, self.all_head_size)
        self.value = nn.Linear(config.d_model, self.all_head_size)

        self.dropout = nn.Dropout(config.dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_head, self.d_head)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def forward(self, q_head, k_head_h, v_head_h,padding_mask=None):

        q_head = self.query(q_head)
        k_head_h = self.key(k_head_h)
        v_head_h = self.value(v_head_h)

        q_layer = self.transpose_for_scores(q_head)
        k_layer = self.transpose_for_scores(k_head_h)
        v_layer = self.transpose_for_scores(v_head_h)

        attn_score = torch.matmul(q_layer,k_layer.transpose(-1,-2))
        attn_score = attn_score / math.sqrt(self.d_head)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2).expand_as(attn_score)
            attn_score = attn_score - 1e30 * padding_mask

        attn_pro = nn.Softmax(dim=-1)(attn_score)
        attn_vec = torch.matmul(attn_pro,v_layer)

        attn_vec = attn_vec.permute(0,2,1,3).contiguous()
        new_attn_vec_shape = attn_vec.size()[:-2] + (self.all_head_size,)
        attn_vec = attn_vec.view(*new_attn_vec_shape)

        outputs = (attn_vec,attn_pro) if self.output_attentions else (attn_vec,)
        return outputs


class VDSelfOutput(nn.Module):

    def __init__(self,config):
        super(VDSelfOutput,self).__init__()
        self.dense = nn.Linear(config.d_model,config.d_model)
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class VDAttention(nn.Module):

    def __init__(self,config):
        super(VDAttention,self).__init__()
        self.self = VDSelfAttention(config)
        self.output = VDSelfOutput(config)
        self.output_attentions = config.output_attentions

    def forward(self,h,padding_mask):

        attn_vec_h = self.self(h,h,h,padding_mask=padding_mask)[0]

        if self.output_attentions:
            attn_vec_h,attn_prob_h = attn_vec_h
        output_h = self.output(attn_vec_h,h)

        if self.output_attentions:
            attn_prob = attn_prob_h
        outputs = (output_h,)
        if self.output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs


class VDIntermediate(nn.Module):

    def __init__(self,config):
        super(VDIntermediate,self).__init__()
        self.dense = nn.Linear(config.d_model,config.intermediate_size)
        if isinstance(config.ff_activation,str):
            self.intermediate_act_fn = ACT2FN[config.ff_activation]
        else:
            self.intermediate_act_fn = config.ff_activation

    def forward(self,h):
        hidden_h = self.dense(h)
        hidden_h = self.intermediate_act_fn(hidden_h)

        return hidden_h


class VDOutput(nn.Module):

    def __init__(self,config):
        super(VDOutput,self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.d_model)
        self.LayerNorm = nn.LayerNorm(config.d_model,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,h,input_h):
        hidden_h = self.dense(h)
        hidden_h = self.dropout(hidden_h)
        hidden_h = self.LayerNorm(hidden_h + input_h)
        return hidden_h


class VDLayer(nn.Module):

    def __init__(self,config):
        super(VDLayer,self).__init__()
        self.attention = VDAttention(config)
        self.intermediate = VDIntermediate(config)
        self.output = VDOutput(config)

    def forward(self,output_h, padding_mask=None):

        attention_outputs = self.attention(output_h,padding_mask)
        attn_ouput_h = attention_outputs[0]
        intermediate_h = self.intermediate(attn_ouput_h)
        output_h = self.output(intermediate_h,attn_ouput_h)

        outputs = (output_h,) + attention_outputs[1:]
        return outputs


class VDPooler(nn.Module):

    def __init__(self,config):
        super(VDPooler,self).__init__()
        self.dense = nn.Linear(config.d_model,config.d_model)
        self.activation = nn.Tanh()
    def forward(self,h):
        h = self.dense(h)
        h = self.activation(h)
        return h

class SentenceEmbedding(nn.Module):

    def __init__(self,d_model,max_seq_len):
        super(SentenceEmbedding, self).__init__()
        pe = torch.zeros(max_seq_len+1,d_model)
        position = torch.arange(0., max_seq_len+1).unsqueeze(1)
        div_term = 1.0 / (20 + torch.exp(torch.arange(0.,d_model,2)*(math.log(10000.0)/d_model)))
        pe[0,:] = 0.0
        pe[:,0::2] = 1/100*torch.sin(position * div_term)
        pe[:,1::2] = 1/100*torch.cos(position * div_term)

        self.register_buffer("pe", pe)
        self.position_embeddings = Variable(pe, requires_grad=False)

    def forward(self, x):
        return self.position_embeddings[x,:]


class VDNetEmbedding(nn.Module):

    def __init__(self,config):
        super(VDNetEmbedding,self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,config.d_model)

        self.img_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.img_loc_embeddings = nn.Linear(5, config.v_hidden_size)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.sentence_emebddings = SentenceEmbedding(config.d_model,config.max_sen_num)

        self.LayerNorm = nn.LayerNorm(config.d_model, eps = config.layer_norm_eps)

    def forward(self,
                input_txt,
                sentence_pos,
                input_img,
                img_loc,
                token_type_ids):

        seq_length = input_txt.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_txt.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_txt).contiguous()

        word_embedings = self.word_embeddings(input_txt)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        sentence_embeddings = self.sentence_emebddings(sentence_pos)

        sentence_embeddings = sentence_embeddings.to(word_embedings.device)
        c_txt_embeddings = word_embedings + position_embeddings + token_type_embeddings + sentence_embeddings

        img_embeddings = self.img_embeddings(input_img)
        img_loc_embeddings = self.img_loc_embeddings(img_loc)
        img_embeddings = img_embeddings + img_loc_embeddings

        c_txt_img_embeddings = torch.cat([c_txt_embeddings,img_embeddings],dim=1)

        output_h = self.dropout(c_txt_img_embeddings)
        output_h = self.LayerNorm(output_h)

        return output_h


class VDNet(nn.Module):

    def __init__(self, config):
        super(VDNet,self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.d_model = config.d_model
        self.n_layer = config.n_layer

        self.embeddings = VDNetEmbedding(config)
        self.layers = nn.ModuleList([VDLayer(config) for _ in range(config.n_layer)])
        self.pooler = VDPooler(config)
        self.dropout = nn.Dropout(config.dropout)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self,new_embeddings):
        self.embedding.word_embeddings = new_embeddings

    def forward(self,
                input_txt,
                sentence_pos,
                input_imgs,
                img_loc,
                token_type_ids,
                padding_mask=None,):

        output_h  = self.embeddings(input_txt,sentence_pos,input_imgs,img_loc,token_type_ids)

        attentions = []
        hidden_states = []
        for i,layer_module in enumerate(self.layers):

            if self.output_hidden_states:
                hidden_states.append(output_h)

            outputs = layer_module(
                output_h,
                padding_mask
            )

            output_h = outputs[0]

            if self.output_attentions:
                attentions.append(outputs[1:])

        if self.output_hidden_states:
            hidden_states.append(output_h)

        output_h= self.pooler(output_h)
        outputs = (output_h,)
        if self.output_hidden_states:
            outputs = outputs + (hidden_states,)
        if self.output_attentions:
            outputs = outputs + (attentions,)
        return outputs


class VDPredictionHeadTransform(nn.Module):

    def __init__(self,config):

        super(VDPredictionHeadTransform,self).__init__()
        self.dense = nn.Linear(config.d_model,config.d_model)
        if isinstance(config.ff_activation,str):
            self.transform_act_fn = ACT2FN[config.ff_activation]
        else:
            self.transform_act_fn = config.ff_activation
        self.LayerNorm = nn.LayerNorm(config.d_model,eps=config.layer_norm_eps)

    def forward(self,h,):
        h = self.dense(h)
        h = self.transform_act_fn(h)
        h = self.LayerNorm(h)
        return h


class VDNetPredictionHead(nn.Module):

    def __init__(self,config):
        super(VDNetPredictionHead,self).__init__()

        self.transform= VDPredictionHeadTransform(config)

        self.lm_classifier = nn.Linear(config.d_model,config.vocab_size)
        self.img_classifier = nn.Linear(config.d_model,config.v_labels)
        self.nsp_classifier = nn.Linear(config.d_model,2)
        self.pos_tag_classifier = nn.Linear(config.d_model,config.pos_num)

        self.lm_transformer = nn.Linear(config.d_model,config.d_model)
        self.pos_tag_transformer = nn.Linear(config.d_model,config.d_model)

    def forward(self,output_h):

        output_h = self.transform(output_h)
        _,length, _ = output_h.shape
        img_hidden = output_h[:,-37:, :]
        nsp_hidden = output_h[:,0, :]
        lm_hidden = output_h[:,:length-37,:]

        lm_hidden = self.lm_transformer(lm_hidden)
        pos_tag_hidden = self.pos_tag_transformer(lm_hidden)

        lm_output = self.lm_classifier(lm_hidden)
        img_output = self.img_classifier(img_hidden)
        nsp_output = self.nsp_classifier(nsp_hidden)
        pos_output = self.pos_tag_classifier(pos_tag_hidden)

        return img_output,lm_output,pos_output,nsp_output

    def get_output_embeddings(self):
        return self.lm_classifier

    def get_pos_out_embeddings(self):
        return self.pos_tag_classifier


class VDModel(nn.Module):

    def __init__(self,config):
        super(VDModel,self).__init__()
        self.config = config
        self.vd_net = VDNet(config)
        self.vd_prediction_head = VDNetPredictionHead(config)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.predict_feature = config.predict_feature

        self.init_weights()
        if self.predict_feature:
            self.vis_criterion = nn.MSELoss(reduction="none")
        else:
            self.vis_criterion = nn.KLDivLoss(reduction="none")


    def init_weights(self):
        self.apply(self._init_weights)
        self._tie_weights()

    def _tie_weights(self):
        def tie_embedings(output_embeddings):
            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = torch.nn.functional.pad(
                    output_embeddings.bias.data,
                    (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],),
                    "constant",
                    0,
                )
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings

        output_embeddings = self.vd_prediction_head.get_output_embeddings()
        input_embeddings = self.vd_net.get_input_embeddings()

        output_embeddings.weight = input_embeddings.weight
        tie_embedings(output_embeddings)

    def _init_weights(self,module):
        """Initialize the weights"""
        if isinstance(module,(nn.Linear,nn.Embedding)):
            if isinstance(module,(nn.Linear,nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module,nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module,torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_txt,sentence_pos,input_img,img_loc,token_type_ids,padding_mask=None,
                lm_label=None,img_label=None,img_target=None,nsp_label=None):

        outputs = self.vd_net(input_txt,sentence_pos,input_img,img_loc,token_type_ids,padding_mask=padding_mask)

        output_h = outputs[0]

        img_out,lm_out,pos_out, nsp_score = self.vd_prediction_head(output_h)
        if lm_label is not None and nsp_label is not None and img_target is not None :
            #compute mask img loss
            if self.predict_feature:
                img_loss = self.vis_criterion(img_out,img_target)
                masked_img_loss = torch.sum(
                    img_loss * (img_label == 1).unsqueeze(2).float()
                ) / max(torch.sum((img_label==1).unsqueeze(2).expand_as(img_loss)),1)
            else:
                img_loss = self.vis_criterion(F.log_softmax(img_out,dim=2),img_target)
                masked_img_loss = torch.sum(
                    img_loss * (img_label == 1).unsqueeze(2).float()
                ) / max(torch.sum((img_label == 1)),0)

            #compute mask language loss
            masked_lm_loss = self.loss_fct(lm_out.view(-1,self.config.vocab_size),
                                           lm_label.view(-1))

            #compute nsp loss
            nsp_loss = self.loss_fct(nsp_score.view(-1,2),nsp_label.view(-1))

            return masked_img_loss.unsqueeze(0), masked_lm_loss.unsqueeze(0), nsp_loss.unsqueeze(0)

        else:
            return nsp_score




if __name__ == "__main__":
    config = Config()
    v = VDModel(config)
    print(v)







