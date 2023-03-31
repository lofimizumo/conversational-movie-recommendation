class MovieResponseConfig():
    def __init__(self, batch_size, max_length, lr, num_epochs, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.max_length = max_length
        self.lr = lr
        self.num_epochs = num_epochs