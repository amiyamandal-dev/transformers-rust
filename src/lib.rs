extern crate tch;

use tch::{nn, Tensor, Device, Kind};
use tch::nn::Init;


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

    pub fn forward(&mut self, values: &Tensor, keys: &Tensor, query: &Tensor, mask: &Tensor) {
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

        let energy = Tensor::einsum("nqhd,nkhd->nhqk", &[query_new_shape, keys_new_shape]);
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


        let mut out = Tensor::einsum("nhql,nlhd->nqbd", &[attention, values_new_shape])
            .reshape(&[N, query_len, self.heads * self.head_dim]);
        /*
        attention shape = (N, heads, query_len, key_len);
        values shape = (N, value_len, self.heads, self.head_dim);

        out shape = (N, query_len, heads, head_dim);
        where key_len == value_len

        then flat last 2 dim
        */

        out = self.fc_out(out);
    }
}






