# An Environment an AI interacts with
#   Author: Bryson Galapon
class Environment(object):
    def perform(self, action):
        raise NotImplementedError

    def getCurrState(self):
        raise NotImplementedError

    def getCurrActions(self):
        raise NotImplementedError
