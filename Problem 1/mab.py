from abc import abstractmethod

class MAB():
    """ Multi Armed Bandit superclass """
    @abstractmethod
    def sample(self):
        """ Call to retrieve action """
        pass