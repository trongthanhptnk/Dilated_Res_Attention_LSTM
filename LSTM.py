import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
import copy


class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_dim, hidden_dim, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_dim, 4 * hidden_dim))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_dim, 4 * hidden_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        init.orthogonal_(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_dim)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        # self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant_(self.bias.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_dim) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_dim).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_prev, c_prev = hx
        batch_size = h_prev.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_prev, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi, self.hidden_dim, dim=1)
        c_1 = torch.sigmoid(f)*c_prev + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_dim}, {hidden_dim})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ResidualLSTMCell(nn.Module):

    """
    A Residual LSTM cell.
    According to the article https://arxiv.org/pdf/1701.03360.pdf.
    """

    def __init__(self, input_dim, hidden_dim, use_bias=True):

        super(ResidualLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        self.weight_xi = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        self.weight_hi = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.weight_ci = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))

        self.weight_xf = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        self.weight_hf = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.weight_cf = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))

        self.weight_xg = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        self.weight_hg = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        
        self.weight_xo = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        self.weight_ho = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.weight_co = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))

        self.weight_c = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.weight_i = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        if use_bias:
            self.bias_i = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.bias_f = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.bias_g = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.bias_o = nn.Parameter(torch.FloatTensor(hidden_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        
        """
        Initialize parameters following the way proposed in the paper.
        """
        
        init.orthogonal_(self.weight_xi.data)
        init.orthogonal_(self.weight_ci.data)
        init.orthogonal_(self.weight_hi.data)
        init.orthogonal_(self.weight_xf.data)
        init.orthogonal_(self.weight_cf.data)
        init.orthogonal_(self.weight_hf.data)
        init.orthogonal_(self.weight_xo.data)
        init.orthogonal_(self.weight_co.data)
        init.orthogonal_(self.weight_ho.data)
        init.orthogonal_(self.weight_xg.data)
        init.orthogonal_(self.weight_hg.data)
        init.orthogonal_(self.weight_c.data)
        init.orthogonal_(self.weight_i.data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant_(self.bias_i.data, val=0)
            init.constant_(self.bias_f.data, val=0)
            init.constant_(self.bias_g.data, val=0)
            init.constant_(self.bias_o.data, val=0)

    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_dim) tensor containing input
                features.
            hx: A tuple (h_prev, c_prev), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_dim).

        Returns:
            h, c: Tensors containing the next hidden and cell state.
        """

        h_prev, c_prev = hx
        i = torch.add(torch.add(torch.add(torch.mm(input_, self.weight_xi), torch.mm(h_prev, self.weight_hi)),
                                torch.mm(c_prev, self.weight_ci)), self.bias_i)
        f = torch.add(torch.add(torch.add(torch.mm(input_, self.weight_xf), torch.mm(h_prev, self.weight_hf)),
                                torch.mm(c_prev, self.weight_cf)), self.bias_f)
        g = torch.add(torch.add(torch.mm(input_, self.weight_xg), torch.mm(h_prev, self.weight_hg)), self.bias_g)
        c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
        o = torch.add(torch.add(torch.add(torch.mm(input_, self.weight_xo), torch.mm(h_prev, self.weight_ho)),
                                torch.mm(c, self.weight_co)), self.bias_o)
        h = torch.tanh(o) * torch.add(torch.mm(torch.tanh(c), self.weight_c), torch.mm(input_, self.weight_i))
        return h, c

    def __repr__(self):
        s = '{name}({input_dim}, {hidden_dim})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LSTM(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_dim, hidden_dim, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(LSTM, self).__init__()
        self.cell_class = cell_class
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            if cell_class in ['LSTM', 'lstm']:
                cell = LSTMCell(input_dim=layer_input_dim, hidden_dim=hidden_dim, **kwargs)
            elif cell_class in ['ResidualLSTM', 'ResLSTM', 'reslstm', 'residuallstm']:
                cell = ResidualLSTMCell(input_dim=layer_input_dim, hidden_dim=hidden_dim, **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        state = []
        for time in range(max_time):
            h_next, c_next = cell(input_=input_[time], hx=hx)
            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask + hx[0]*(1 - mask)
            c_next = c_next*mask + hx[1]*(1 - mask)
            hx_next = (h_next, c_next)
            output.append(h_next)
            state.append(c_next)
            hx = hx_next
        output = torch.stack(output, 0)
        state = torch.stack(state, 0)
        return output, state, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                device = input_.get_device()
                length = length.cuda(device)
        if hx is None:
            hx_0 = torch.empty(self.num_layers, batch_size, self.hidden_dim)
            hx_0 = nn.init.xavier_uniform_(hx_0, gain=nn.init.calculate_gain('relu'))
            hx_1 = torch.empty(self.num_layers, batch_size, self.hidden_dim)
            hx_1 = nn.init.xavier_uniform_(hx_1, gain=nn.init.calculate_gain('relu'))
            hx = (Variable(hx_0), Variable(hx_1))
        h_n = []
        c_n = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            hx_layer = (hx[0][layer,:,:], hx[1][layer,:,:])
            
            if layer == 0:
                layer_output, layer_state, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=input_, length=length, hx=hx_layer)
            else:
                layer_output, layer_state, (layer_h_n, layer_c_n) = LSTM._forward_rnn(
                    cell=cell, input_=layer_output, length=length, hx=hx_layer)
            
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        state = layer_state
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        output = torch.cat((hx_0, output), dim=0)
        state = torch.cat((hx_1, state), dim=0)
        return output, state, (h_n, c_n)


class DilatedRNN(nn.Module):

    def __init__(self, hidden_dims, cells, dilations, n_classes=1, input_dim=1):
        super(DilatedRNN, self).__init__()
        assert (len(cells) == len(dilations))
        assert (len(cells) == len(hidden_dims))

        self.hidden_dims = hidden_dims
        self.cells = cells
        self.dilations = dilations
        self.input_dim = input_dim

        self.linear = nn.Linear(hidden_dims[-1], n_classes)

    def dRNN(self, inputs, hidden_dim, cell, dilation):

        """
        This function constructs a layer of dilated RNN. Xây dựng một lớp layer trong kiến trúc chung.
        Inputs:
            cell -- the dilation operations is implemented independently of the RNN cell.
                In theory, any valid tensorflow rnn cell should work.
            inputs -- the input for the RNN. inputs should be in the form of
                a list of 'n_steps' tensors. Each has shape (batch_size, input_dim).
                (T, batch_size, input_dim).
            dilation -- the dilation here refers to the 'dilations' in the orginal WaveNet paper.
            scope -- variable scope.
        Outputs:
            outputs -- the outputs from the RNN.
        """

        n_steps = len(inputs)
        if dilation < 0 or dilation >= n_steps:
            raise ValueError('The \'dilation\' variable needs to be adjusted.')

        # Make the length of inputs divide 'dilation', by using zero-padding.
        EVEN = (n_steps % dilation) == 0

        if not EVEN:

            """
            Create a tensor in shape (batch_size, input_dim), which all elements are zero.  
            This is used for zero padding.
            """

            zero_tensor = torch.zeros_like(inputs[0])
            dilated_n_steps = n_steps // dilation + 1
            input_list = list(inputs)
            for _ in range(dilated_n_steps * dilation - n_steps):
                input_list.append(zero_tensor)
            inputs = torch.stack(input_list, 0)

        else:
            dilated_n_steps = n_steps // dilation

        """
        now the length of 'inputs' divide dilation
        reshape it in the format of a list of tensors
        the length of the list is 'dilated_n_steps' 
        the shape of each tensor is [batch_size * dilation, input_dim] 
        by stacking tensors that "colored" the same

        Example: 
        n_steps is 5, dilation is 2, inputs = [x1, x2, x3, x4, x5]
        zero-padding --> [x1, x2, x3, x4, x5, 0]
        we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
        which the length is the ceiling of n_steps/dilation
        """

        dilated_inputs = torch.stack(
            [torch.cat(tuple(inputs[i * dilation: (i + 1) * dilation]), dim=0) for i in range(dilated_n_steps)], dim=0)
        # dilated inputs bây giờ là gộp dilation phần tử.

        # building a dilated RNN with reformated (dilated) inputs:
        input_dim = inputs[0].size(1)
        if cell == 'RNN':
            model = nn.RNN(input_dim, hidden_dim)
        elif cell == 'LSTM':
            model = LSTM('LSTM', input_dim, hidden_dim)
        elif cell == 'GRU':
            model = nn.GRU(input_dim, hidden_dim)
        elif cell == 'ResidualLSTM':
            model = LSTM('ResidualLSTM', input_dim, hidden_dim)
        else:
            print('Not support.')

        dilated_outputs, states, _ = model(dilated_inputs)
        dilated_outputs = dilated_outputs[1:]

        """
        reshape output back to the input format as a list of tensors with shape [batch_size, input_dim]
        split each element of the outputs from size [batch_size*dilation, input_dim] to
        [[batch_size, input_dim], [batch_size, input_dim], ...] with length = dilation
        """

        splitted_outputs = []
        for i in range(dilated_n_steps):
            for j in range(dilation):
                splitted_outputs.append(dilated_outputs[i, j].unsqueeze(0))
        unrolled_outputs = torch.stack(splitted_outputs, 0)
        outputs = unrolled_outputs[:n_steps]    # Remove padded zeros.

        return outputs

    def multi_dRNN_with_dilations(self, inputs):

        """
        This function constucts a multi-layer dilated RNN.
        Inputs:
            cells -- A list of RNN cells.
            inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dim).
            dilations -- A list of integers with the same length of 'cells' indicates the dilations for each layer.
        Outputs:
            x -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
        """

        x = copy.copy(inputs)
        for i in range(len(self.cells)):
            x = self.dRNN(x, self.hidden_dims[i], self.cells[i], self.dilations[i])
        return x

    def forward(self, inputs):
        """
        inputs -- the input for the RNN. inputs should be in the form of
            a list of 'n_steps' tenosrs. Each has shape (batch_size, input_dim)
        """

        layer_outputs = self.multi_dRNN_with_dilations(inputs)
        prediction = self.linear(layer_outputs[-1]).float()

        return prediction


class AttentionLSTM(nn.Module):

    """
    Follow the article https://arxiv.org/pdf/1704.02971.pdf.

    Input_ --  n Time Series, each of length T. Must be a tensor of size (n, 1, T).
    Driving_input_ --  T series, each of length T (transpose of Input_). A tensor of size (T, n).
    Target --  1 Time Series of length T-1, prediction is the next value of target. Must be a tensor of size (1, 1, T-1).
    
    Input_dim --  T
    Driving_dim --  n
    Encoder_dim --  m
    Decoder_dim --  p

    weight_e, u_e, v_e --  equation (8) in the article.
    weight_d, u_d, v_d --  equation (12) in the article.
    weight_tidle, bias_tidle --  equation (15) in the article.
    weight_out, v_out, bias_out, bias_v --  equation (22) in the article.

    InputAttLayer --  Input Attention Layer, which is a LSTM Layer.
    Encoder --  also a LSTM Layer, but runs cell-by-cell, thus we define it by a LSTMCell.
    Decoder --  similar to Encoder.
    """

    def __init__(self, input_dim, driving_dim, encoder_dim, decoder_dim, use_bias=True, encoder_type = 'LSTM',
                 decoder_type = 'LSTM'):

        super(AttentionLSTM, self).__init__()
        # assert (target.size(2) == input_.size(2) - 1)
        # self.input_ = input_
        # self.driving_input_ = input_.transpose(0, 2).squeeze(1)
        # 
        # self.target = target
        
        self.input_dim = input_dim
        self.driving_dim = driving_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.use_bias = use_bias

        self.weight_e = nn.Parameter(torch.FloatTensor(2 * encoder_dim, self.input_dim))
        self.u_e = nn.Parameter(torch.FloatTensor(self.input_dim, self.input_dim))
        self.v_e = nn.Parameter(torch.FloatTensor(self.input_dim, 1))

        self.Encoder = LSTM(encoder_type, self.driving_dim, encoder_dim)

        self.weight_d = nn.Parameter(torch.FloatTensor(2 * decoder_dim, encoder_dim))
        self.u_d = nn.Parameter(torch.FloatTensor(encoder_dim, encoder_dim))
        self.v_d = nn.Parameter(torch.FloatTensor(encoder_dim, 1))

        self.weight_tidle = nn.Parameter(torch.FloatTensor(encoder_dim + 1, 1))

        if decoder_type == 'LSTM':
            self.Decoder = LSTMCell(1, self.decoder_dim)
        elif decoder_type == 'ResidualLSTM':
            self.Decoder = ResidualLSTMCell(1, self.decoder_dim)

        self.weight_out = nn.Parameter(torch.FloatTensor(encoder_dim + decoder_dim, decoder_dim))
        self.v_out = nn.Parameter(torch.FloatTensor(decoder_dim, 1))

        if use_bias:
            self.bias_tidle = nn.Parameter(torch.FloatTensor(1, 1))
            self.bias_out = nn.Parameter(torch.FloatTensor(1, decoder_dim))
            self.bias_v = nn.Parameter(torch.FloatTensor(1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        init.orthogonal_(self.weight_e.data)
        init.orthogonal_(self.weight_d.data)
        init.orthogonal_(self.weight_tidle.data)
        init.orthogonal_(self.weight_out.data)
        init.orthogonal_(self.u_e.data)
        init.orthogonal_(self.u_d.data)
        init.orthogonal_(self.v_e.data)
        init.orthogonal_(self.v_d.data)
        init.orthogonal_(self.v_out.data)
        # The bias is just set to zero vectors.
        if self.use_bias:
            init.constant_(self.bias_tidle.data, val=0)
            init.constant_(self.bias_out.data, val=0)
            init.constant_(self.bias_v.data, val=0)

    def make_matrix(self, tensor, size):

        """
        Given a vector (size (1, length) or (length, 1)), create a matrix of which all rows or columns are that vector.
        """

        if tensor.size(1) == 1:
            matrix = tensor
            for _ in range(size-1):
                matrix = torch.cat((matrix, tensor), dim=1)
        elif tensor.size(0) == 1:
            matrix = tensor
            for _ in range(size - 1):
                matrix = torch.cat((matrix, tensor), dim=0)
        return matrix

    def TempAttLayer(self, inputs, d_prev, s_prev):

        """
        Temporal Attention Layer, simply a calculation layer.
        In each step, it takes in d_prev and s_prev of the Decoder and produces a new context.

        inputs -- of size (T, 1, m).
        d_prev -- of size (1, p).
        s_prev --  of size (1, p).
        ds -- of size (1, 2p).
        ds_matrix --  matrix of which all rows are ds, of size (T, 2p).
        weight_d --  of size (2p, m).
        u_d --  of size (m, m).
        v_d --  of size (m, 1).
        l --  of size (T, 1).
        beta --  of size (T, 1).
        context --  of size (1, m).
        """

        ds = torch.cat((d_prev, s_prev), dim=1)
        ds_matrix = self.make_matrix(ds, self.input_dim)
        l = torch.mm(
            torch.tanh(torch.add(torch.mm(ds_matrix, self.weight_d), torch.mm(inputs.squeeze(1), self.u_d))), self.v_d)
        beta = functional.softmax(l, dim=0)
        context = torch.zeros(1, self.encoder_dim)
        for i in range(self.input_dim):
            context = torch.add(context, beta[i] * inputs[i])
        return context

    def forward(self, input_, target):

        """
        Input Attention Layer --  takes in T time series, produces T series with score coefficients.

        driving_input_ --  (T, n), unsqueeze(1) to be of size (T, 1, n).
        outputs_InputAttLayer, states_InputAttLayer --  (T, 1, m).
        hs --  concatenation of outputs_InputAttLayer and states_InputAttLayer, of size (T, 1, 2m).
        hs_matrix --  matrix of which all rows are hs[t], of size (n, 2m).
        weight_e --  of size (2m, T).
        input_.squeeze(1) --  of size (n, T).
        u_e --  of size (T, T).
        v_e --  of size (T, 1).
        e --  of size (n, 1).
        alpha --  of size (1, n).
        driving_input_.unsqueeze(1)[t] --  of size (1, n).
        encoder_inputs --  of size (T, 1, n).
        """

        assert (target.size(2) == input_.size(2) - 1)
        driving_input_ = input_.transpose(0, 2).squeeze(1)

        outputs_InputAttLayer, states_InputAttLayer, _ = self.Encoder.forward(driving_input_.unsqueeze(1))
        hs = torch.cat((outputs_InputAttLayer[:-1], states_InputAttLayer[:-1]), 2)
        encoder_inputs = torch.zeros(1, 1, self.driving_dim)
        for t in range(self.input_dim):
            hs_matrix = self.make_matrix(hs[t], self.driving_dim)
            e = torch.mm(torch.tanh(torch.add(torch.mm(hs_matrix, self.weight_e), torch.mm(input_.squeeze(1), self.u_e))), self.v_e)
            alpha = functional.softmax(e, dim=0).transpose(0, 1)
            encoder_input = alpha * driving_input_.unsqueeze(1)[t]
            encoder_inputs = torch.cat((encoder_inputs, encoder_input.unsqueeze(0)), dim=0)
        encoder_inputs = encoder_inputs[1:]

        """
        Encoder: takes in encoder_inputs (T series of size n) and produces T tensors of size (1, encoder_dim).
        
        outputs_Encoder --  of size (T, 1, m).
        """

        outputs_Encoder, _, _ = self.Encoder.forward(encoder_inputs)
        outputs_Encoder = outputs_Encoder[1:]

        """
        Temporal Attention Layer to Decoder.
        
        d_prev --  of size (1, p).
        s_prev --  of size (1, p).
        context --  T contexts, each of size (1, m).
        yc --  concatenation of context and target[t], of size (1, m + 1).
        weight_tidle --  of size (m + 1, 1).
        bias_tidle --  of size (1, 1).
        decode_input --  of size (1, 1).
        """

        d_prev = torch.zeros(1, self.decoder_dim)
        s_prev = torch.zeros(1, self.decoder_dim)
        for t in range(self.input_dim):
            if t < self.input_dim - 1:
                # Temporal Attention Layer
                context = self.TempAttLayer(outputs_Encoder, d_prev, s_prev)
                yc = torch.cat((target.transpose(0, 2)[t], context), dim=1)
                decoder_input = torch.add(torch.mm(yc, self.weight_tidle), self.bias_tidle)
                d_prev, s_prev = self.Decoder(decoder_input, (d_prev, s_prev))
            elif t == self.input_dim - 1:
                # Temporal Attention Layer
                context = self.TempAttLayer(outputs_Encoder, d_prev, s_prev)

        """
        Final step to make prediction.
        
        dc --  of size (1, p + m).
        weight_out --  of size (p + m, p).
        bias_out --  of size (1, p).
        v_out --  of size (p, 1).
        bias_v --  of size (1, 1).
        prediction --  of size (1, 1)
        """

        dc = torch.cat((d_prev, context), dim=1)
        prediction = torch.add(torch.mm(torch.add(torch.mm(dc, self.weight_out), self.bias_out), self.v_out), self.bias_v)
        
        return prediction

    def __repr__(self):
        s = '{name}({driving_dim}, {hidden_dim})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

