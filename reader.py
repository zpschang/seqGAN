import re
EOS_ID = 1
UNK_ID = 2
class reader():
    def __init__(self, file_name_post, file_name_resp, file_name_word):
        with open(file_name_word, 'rb') as file_word:
            self.d = {}
            self.symbol = []
            num = 0
            for line in file_word.readlines():
                line = line.decode('utf-8')[:-1]
                self.symbol.append(line)
                self.d[line] = num
                num += 1
        self.file_name_post = file_name_post
        self.file_name_resp = file_name_resp
        self.post = open(self.file_name_post, 'rb')
        self.resp = open(self.file_name_resp, 'rb')
        self.epoch = 0
        self.k = 0

    def get_batch(self, batch_size):
        result = []
        self.k += batch_size
        for _ in range(batch_size):
            post = self.post.readline()
            resp = self.resp.readline()
            if not post:
                self.restore()
                self.epoch += 1
                self.k = 0
                print 'epoch: ', self.epoch
                return self.get_batch(batch_size)
            post = post.decode('utf-8')[:-1]
            resp = resp.decode('utf-8')[:-1]
            words_post = re.split(' ', post)
            words_resp = re.split(' ', resp)
            index_post = [self.d[word] if word in self.d else UNK_ID for word in words_post]
            index_resp = [self.d[word] if word in self.d else UNK_ID for word in words_resp]
            index_resp = index_resp + [EOS_ID]
            result.append((index_post, index_resp))
        return result

    def restore(self):
        self.post.close()
        self.resp.close()
        self.post = open(self.file_name_post, 'rb')
        self.resp = open(self.file_name_resp, 'rb')
