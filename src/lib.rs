extern crate tch;

use tch::{Device, Kind, nn, Tensor};
use tch::nn::{Init, Module, ModuleT};


#[derive(Debug)]
pub struct SelfAttention {
    device: Device,
    embed_size: i64,
    heads: i64,
    head_dim: i64,
    values: nn::Linear,
    keys: nn::Linear,
    queries: nn::Linear,
    fc_out: nn::Linear,
}

impl SelfAttention {
    pub fn new(vs: nn::Path, embed_size: i64, heads: i64) -> Self {
        let head_dim = embed_size / heads;
        assert_eq!(head_dim * heads, embed_size);
        let config = nn::LinearConfig {
            ws_init: Init::KaimingUniform,
            bs_init: None,
            bias: false,
        };
        SelfAttention {
            device: vs.device(),
            embed_size: embed_size,
            heads: heads,
            head_dim: head_dim,
            values: nn::linear(&vs, head_dim, head_dim, config),
            keys: nn::linear(&vs, head_dim, head_dim, config),
            queries: nn::linear(&vs, head_dim, head_dim, config),
            fc_out: nn::linear(&vs, head_dim * heads, embed_size, Default::default()),
        }
    }

    pub fn forward(&mut self, values: &Tensor, keys: &Tensor, query: &Tensor, mask: &Tensor) -> Tensor {
        let mut temp = query.size();
        let N = temp[0];
        let query_len = temp[1];
        temp = values.size();
        let value_len = temp[1];
        temp = keys.size();
        let key_len = temp[1];

        let values_new_shape = values.reshape(&[N, value_len, self.heads, self.head_dim]);
        let keys_new_shape = keys.reshape(&[N, key_len, self.heads, self.head_dim]);
        let query_new_shape = query.reshape(&[N, query_len, self.heads, self.head_dim]);

        let values_new = self.values.forward_t(&values_new_shape, true);
        let keys_new = self.keys.forward_t(&keys_new_shape, true);
        let query_new = self.queries.forward_t(&query_new_shape, true);


        let energy = Tensor::einsum("nqhd,nkhd->nhqk", &[query_new, keys_new]);
        /*
        query_new_shape :(N, query_len, self.heads, self.head_dim)
        keys_new_shape :(N, key_len, self.heads, self.head_dim)

        energy :(N, heads, query_len, key_len);
        */

        // todo:- mask condition
        /*
        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))
        */

        let mut attention = energy / self.embed_size.pow(1 / 2);
        attention = attention.softmax(3, Kind::Float);


        let mut out = Tensor::einsum("nhql,nlhd->nqbd", &[attention, values_new])
            .reshape(&[N, query_len, self.heads * self.head_dim]);
        /*
        attention shape = (N, heads, query_len, key_len);
        values shape = (N, value_len, self.heads, self.head_dim);

        out shape = (N, query_len, heads, head_dim);
        where key_len == value_len

        then flat last 2 dim
        */

        out = self.fc_out.forward_t(&out, true);
        out
    }
}

#[derive(Debug)]
pub struct TransformerBlock {
    attention: SelfAttention,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    liner1: nn::Linear,
    liner2: nn::Linear,
    dropout: f64,
}


impl TransformerBlock {
    pub fn new(vs: nn::Path, embed_size: i64, heads: i64, dropout: f64, forward_expansion: i64) -> Self {
        let norm1 = nn::layer_norm(&vs, vec![embed_size], Default::default());
        let norm2 = nn::layer_norm(&vs, vec![embed_size], Default::default());
        let liner1 = nn::linear(&vs, embed_size, forward_expansion * embed_size, Default::default());
        let liner2 = nn::linear(&vs, forward_expansion * embed_size, embed_size, Default::default());
        let attention = SelfAttention::new(vs, embed_size.clone(), heads.clone());


        TransformerBlock {
            attention,
            norm1,
            norm2,
            liner1,
            liner2,
            dropout,
        }
    }


    pub fn forward(&mut self, values: &Tensor, keys: &Tensor, query: &Tensor, mask: &Tensor) -> Tensor {
        let attention = self.attention.forward(values, keys, query, mask);
        let x = self.norm1.forward_t(&(attention + values), true);
        // feed forwoard network
        let x = self.liner1.forward_t(&x.dropout(self.dropout, false), true);
        let x = self.liner2.forward_t(&x.relu(), true);

        let x = self.norm2.forward_t(&x, true);
        x.dropout(self.dropout, false)
    }
}

#[derive(Debug)]
pub struct Encoder {
    embed_size: i64,
    word_embedding: nn::Embedding,
    position_embedding: nn::Embedding,
    layers: Vec<TransformerBlock>,
}

impl Encoder {
    pub fn new(src_vocab_size: i64,
               embed_size: i64,
               num_layers: i64,
               heads: i64,
               vs: nn::Path,
               forward_expansion: i64,
               dropout: f64,
               max_length: i64) {
        let word_embedding = nn::embedding(&vs, src_vocab_size, embed_size, Default::default());
        let position_embedding = nn::embedding(&vs, max_length, embed_size,Default::default());
        let mut temp_vec :Vec<TransformerBlock> = vec![];
        for i in 0..num_layers{

            let t = TransformerBlock::new(vs, embed_size,heads, dropout, forward_expansion);
            temp_vec.push(t);

        }
    }
}



