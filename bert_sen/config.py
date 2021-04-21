class Config(object):
    def __init__(
        self,
        output_attentions=False,
        output_hidden_states=False,
        n_head=16,
        d_model=768,
        vocab_size=30522,
        intermediate_size=3072,
        dropout=0.1,
        layer_norm_eps=1e-12,
        d_inner=4096,
        ff_activation="gelu",
        max_position_embeddings=512,
        v_feature_size=2048,
        v_hidden_size=768,
        type_vocab_size=2,
        n_layer=12,
        initializer_range=0.02,
        v_labels=1601,
        pos_num=33,
        max_sen_num=21,
        predict_feature=False,
    ):

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.n_head = n_head
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.d_inner = d_inner
        self.ff_activation = ff_activation
        self.max_position_embeddings = max_position_embeddings
        self.v_feature_size = v_feature_size
        self.v_hidden_size = v_hidden_size
        self.type_vocab_size = type_vocab_size
        self.n_layer = n_layer
        self.initializer_range = initializer_range
        self.v_labels = v_labels
        self.predict_feature = predict_feature
        self.intermediate_size = intermediate_size
        self.pos_num = 33
        self.max_sen_num = 20
