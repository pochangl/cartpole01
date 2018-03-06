'''
    Draph is TensorFlow graph management Class
'''

class Graph:
    '''
        graph class for manipulating TensorFlow graph
    '''
    iput = None
    pre_iput = None
    oput = None
    build_decision = None
    trainers = None
    ipt = None

    def __init__(self):
        self.build()

    def build_preprocess_graph(self):
        '''
            process input for better manipulation
        '''
        return self.ipt

    def build_graph(self, iput):
        '''
            build main logic
        '''
        return iput

    def build_decision_graph(self, iput, oput):
        '''
            making decision
        '''
        return iput

    def build_trainers(self, *args, **kwargs):
        '''
            get trains
        '''
        raise NotImplementedError()

    def build(self):
        '''
            build the overall graph
        '''
        self.pre_iput = pre_iput = self.build_preprocess_graph()
        self.oput = oput = self.build_graph(iput=pre_iput)
        self.oput_
        self.build_decision = decision = self.build_decision_graph(iput=pre_iput, oput=oput)
        self.trainers = self.build_trainers(iput=pre_iput, oput=oput, decision=decision)
