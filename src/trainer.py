"""Wrap the training functions of pc and rnn

original code found at: https://xylambda.github.io/blog/python/pytorch/machine-learning/2021/01/04/pytorch_trainer.html
"""
#collapse
import time
import logging
import warnings
import torch

class BaseTrainer:
    """Trainer
    
    Base class for trainers for RNN and PCN
    
    Parameters
    ----------
    seq_len : int
        Length of the training sequence
    model : torch.Module
        The model to train.
    criterion : torch.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    step_update : bool
        Whether to update weights at each step; for PC this this always true
    logger_kwards : dict
        Args for ..
        
    Attributes
    ----------
    train_loss_ : list
    val_loss_ : list
    
    """
    def __init__(
        self, 
        seq_len,
        model, 
        criterion, 
        optimizer, 
        step_update,
        logger_kwargs, 
        device=None
    ):
        self.seq_len = seq_len
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.step_update = step_update
        self.logger_kwargs = logger_kwargs
        self.device = self._get_device(device)
        
        # send model to device
        self.model.to(self.device)

        # attributes        
        self.train_loss_ = []
        self.val_loss_ = []

        logging.basicConfig(level=logging.INFO, format='%(message)s')
        
    def fit(self, train_loader, val_loader, epochs, bsz):
        """Fits.
        
        Fit the model using the given loaders for the given number
        of epochs.
        
        Parameters
        ----------
        train_loader : 
        val_loader : 
        epochs : int
            Number of training epochs.
        bsz : int
            Batch size
        
        """
        # number of batches
        nbatches_tr = len(train_loader)
        nbatches_val = len(val_loader)

        # track total training time
        total_start_time = time.time()

        # ---- train process ----
        for epoch in range(epochs):
            # track epoch time
            epoch_start_time = time.time()

            # training 
            self.model.train()
            tr_loss = 0
            for features, labels in train_loader:
                # train loss of one batch
                tr_loss += self._train_on_batch(features, labels, bsz)
            tr_loss /= nbatches_tr
            
            # validate
            self.model.eval()
            val_loss = 0
            for features, labels in train_loader:
                # val loss of one batch
                val_loss = self._validate_on_batch(features, labels, bsz)
            val_loss /= nbatches_val
            
            self.train_loss_.append(tr_loss.item())
            self.val_loss_.append(val_loss.item())

            epoch_time = time.time() - epoch_start_time
            self._logger(
                tr_loss, 
                val_loss, 
                epoch+1, 
                epochs, 
                epoch_time, 
                **self.logger_kwargs
            )

        total_time = time.time() - total_start_time

        # final message
        logging.info(
            f"""End of training. Total time: {round(total_time, 5)} seconds"""
        )

    def predict(self, features, labels, bsz):
        with torch.no_grad():
            # move to device
            features, labels = self._to_device(features, labels, self.device)

            # initialize hidden activities of rnn
            h = self.model.init_hidden(bsz).to(self.device)
                
            # iterate through the time steps of batched seqs
            pred = []
            for seq_idx in range(self.seq_len):

                # forward pass on each individual time step
                h, out = self.model(features[:, seq_idx:seq_idx+1], h)
                
                pred.append(out)

            pred = torch.cat(pred, dim=1)
                
        return pred
        
    def _logger(
        self, 
        tr_loss, 
        val_loss, 
        epoch, 
        epochs, 
        epoch_time, 
        show=True, 
        update_step=20
    ):
        if show:
            if epoch % update_step == 0 or epoch == 1:
                # to satisfy pep8 common limit of characters
                msg = f"Epoch {epoch}/{epochs} | Train loss: {tr_loss}" 
                msg = f"{msg} | Validation loss: {val_loss}"
                msg = f"{msg} | Time/epoch: {round(epoch_time, 5)} seconds"

                logging.info(msg)
    
    def _train_on_batch(self, features, labels, bsz):        

        raise NotImplementedError()
    
    def _validate_on_batch(self, features, labels, bsz):
        
        raise NotImplementedError()
    
    def _to_device(self, features, labels, device):
        return features.to(device), labels.to(device)
    
    def _compute_loss(self, real, target):
        try:
            loss = self.criterion(real, target)
        except:
            loss = self.criterion(real, target.long())
            msg = f"Target tensor has been casted from"
            msg = f"{msg} {type(target)} to 'long' dtype to avoid errors."
            warnings.warn(msg)

        # apply regularization if any
        # loss += penalty.item()
            
        return loss

    def _get_device(self, device):
        if device is None:
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {dev}"
            warnings.warn(msg)
        else:
            dev = device

        return dev

class RNNTrainer(BaseTrainer):
    """RNN Trainer
    
    Parameters
    ----------
    Inherited base trainer parameters
        
    Attributes
    ----------
    train_loss_ : list
    val_loss_ : list
    
    """
    def __init__(
        self, 
        seq_len,
        model, 
        criterion, 
        optimizer, 
        step_update,
        logger_kwargs, 
        device=None
    ):
        super().__init__(seq_len,
                         model, 
                         criterion, 
                         optimizer, 
                         step_update,
                         logger_kwargs, 
                         device=device)
    
    def _train_on_batch(self, features, labels, bsz):        
        # move to device
        features, labels = self._to_device(features, labels, self.device)

        # initialize hidden activities of rnn
        h = self.model.init_hidden(bsz).to(self.device)

        # iterate through the time steps of batched seqs
        batch_loss = 0
        for seq_idx in range(self.seq_len):

            # forward pass on each individual time step
            h, out = self.model(features[:, seq_idx:seq_idx+1], h)
            
            # loss
            loss = self._compute_loss(out, labels[:, seq_idx:seq_idx+1])
            
            if self.step_update:
                # remove gradient from previous passes
                self.optimizer.zero_grad()
                
                # backprop
                loss.backward()
                
                # parameters update
                self.optimizer.step()

            batch_loss += loss

        # average across seq
        batch_loss /= self.seq_len

        """In cases where we update weigths after the whole sequence,
        
        accumulate the losses in a batch and then backprop from the accumulated loss
        """
        if not self.step_update:
            # remove gradient from previous passes
            self.optimizer.zero_grad()
            
            # backprop
            batch_loss.backward()
            
            # parameters update
            self.optimizer.step()
            
        return batch_loss
    
    def _validate_on_batch(self, features, labels, bsz):
        
        with torch.no_grad():
            # move to device
            features, labels = self._to_device(features, labels, self.device)

            # initialize hidden activities of rnn
            h = self.model.init_hidden(bsz).to(self.device)
                
            # iterate through the time steps of batched seqs
            batch_loss = 0
            for seq_idx in range(self.seq_len):

                # forward pass on each individual time step
                h, out = self.model(features[:, seq_idx:seq_idx+1], h)
                
                # loss
                loss = self._compute_loss(out, labels[:, seq_idx:seq_idx+1])
                batch_loss += loss

            # average across seq
            batch_loss /= self.seq_len
                
        return batch_loss
    
class PCTrainer(BaseTrainer):
    """tPCN Trainer
    
    Parameters
    ----------
    Inherited base trainer parameters

    inference_kwargs : dict
        Args for running inference in PCns
        
    Attributes
    ----------
    train_loss_ : list
    val_loss_ : list
    
    """
    def __init__(
        self, 
        seq_len,
        model, 
        criterion, 
        optimizer, 
        step_update,
        logger_kwargs, 
        inference_kwargs,
        device=None
    ):
        super().__init__(seq_len,
                         model, 
                         criterion, 
                         optimizer, 
                         step_update,
                         logger_kwargs, 
                         device=device)
        self.inference_kwargs = inference_kwargs

    def predict(self, features, labels, bsz):
        # get the inference hyperparameters
        inf_iters = self.inference_kwargs['inf_iters']
        inf_lr = self.inference_kwargs['inf_lr']
        
        with torch.no_grad():
            # move to device
            features, labels = self._to_device(features, labels, self.device)

            # initialize hidden activities of rnn
            h = self.model.init_hidden(bsz).to(self.device)
                
            # iterate through the time steps of batched seqs
            pred = []
            for seq_idx in range(self.seq_len):

                # pc inference step for the hidden state
                self.model.inference(inf_iters, 
                                     inf_lr, 
                                     labels[:, seq_idx:seq_idx+1], 
                                     features[:, seq_idx:seq_idx+1], 
                                     h)
                
                # assign the current hidden state to h
                h = self.model.z.clone()
                
                pred.append(self.model.pred_x)
                
        return torch.cat(pred, dim=1)

    def _train_on_batch(self, features, labels, bsz):        
        # get the inference hyperparameters
        inf_iters = self.inference_kwargs['inf_iters']
        inf_lr = self.inference_kwargs['inf_lr']

        # move to device
        features, labels = self._to_device(features, labels, self.device)

        # initialize hidden activities of rnn
        h = self.model.init_hidden(bsz).to(self.device)

        # iterate through the time steps of batched seqs
        batch_loss = 0
        # observation loss, to compare with RNN
        obs_loss = 0
        for seq_idx in range(self.seq_len):
            # remove gradient from previous passes
            self.optimizer.zero_grad()

            # pc inference step for the hidden state
            self.model.inference(inf_iters, 
                                 inf_lr, 
                                 labels[:, seq_idx:seq_idx+1], 
                                 features[:, seq_idx:seq_idx+1], 
                                 h)
            # loss for PCN is not the observations loss
            # it is the total energy of the model
            loss = self.model.get_loss(labels[:, seq_idx:seq_idx+1], 
                                       features[:, seq_idx:seq_idx+1], 
                                       h)
            
            # backprop
            loss.backward()
            
            # parameters update
            self.optimizer.step()

            # assign h to the current hidden state
            h = self.model.z.clone()

            batch_loss += loss
            obs_loss += self._compute_loss(self.model.pred_x,
                                           labels[:, seq_idx:seq_idx+1])

        # average across seq
        batch_loss /= self.seq_len
        obs_loss /= self.seq_len

        return obs_loss
    
    def _validate_on_batch(self, features, labels, bsz):
        # get the inference hyperparameters
        inf_iters = self.inference_kwargs['inf_iters']
        inf_lr = self.inference_kwargs['inf_lr']
        
        with torch.no_grad():
            # move to device
            features, labels = self._to_device(features, labels, self.device)

            # initialize hidden activities of rnn
            h = self.model.init_hidden(bsz).to(self.device)
                
            # iterate through the time steps of batched seqs
            batch_loss = 0
            obs_loss = 0
            for seq_idx in range(self.seq_len):

                # pc inference step for the hidden state
                self.model.inference(inf_iters, 
                                     inf_lr, 
                                     labels[:, seq_idx:seq_idx+1], 
                                     features[:, seq_idx:seq_idx+1], 
                                     h)
                
                # loss
                loss = self.model.get_loss(labels[:, seq_idx:seq_idx+1], 
                                           features[:, seq_idx:seq_idx+1], 
                                           h)
                
                # assign the current hidden state to h
                h = self.model.z.clone()
                
                batch_loss += loss
                obs_loss += self._compute_loss(self.model.pred_x,
                                               labels[:, seq_idx:seq_idx+1])

            # average across seq
            batch_loss /= self.seq_len
            obs_loss /= self.seq_len
                
        return obs_loss