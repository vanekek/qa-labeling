class PipeLineConfig:
    def __init__(self, lr, warmup,accum_steps, epochs, seed, expname,head_tail,freeze,question_weight,answer_weight,fold,train):
        self.lr = lr
        self.warmup = warmup
        self.accum_steps = accum_steps
        self.epochs = epochs
        self.seed = seed
        self.expname = expname
        self.head_tail = head_tail
        self.freeze = freeze
        self.question_weight = question_weight
        self.answer_weight =answer_weight
        self.fold = fold
        self.train = train