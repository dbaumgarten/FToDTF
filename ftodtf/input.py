"""This module handles all the input-relatet tasks like loading, pre-processing and batching"""
import os
import urllib
import zipfile
import collections
import random
from tempfile import gettempdir

def maybe_download(url, filename, expected_bytes, unzip=True):
  """Download a file if not present, and make sure it's the right size."""
  local_filename = os.path.join(gettempdir(), filename)
  if not os.path.exists(local_filename):
    local_filename, _ = urllib.request.urlretrieve(url + filename,
                                                   local_filename)
  statinfo = os.stat(local_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + local_filename +
                    '. Can you get to it with a browser?')

  if unzip:
    zip_ref = zipfile.ZipFile(local_filename, 'r')
    local_filename = os.path.join(gettempdir(),zip_ref.namelist()[0])
    if not os.path.exists(local_filename):
      print("Unpacking file...")
      zip_ref.extract(zip_ref.namelist()[0],path=gettempdir())
    zip_ref.close()

  return local_filename

class InputProcessor():
  """Handles the creation of training-examble-batches from the raw training-text"""
  def __init__(self,filename,skip_window,batch_size,vocab_size):
    self.filename = filename
    self.skip_window = skip_window
    self.batch_size = batch_size
    self.vocab_size = vocab_size

  def preprocess(self):
    """ Do the needed proprocessing of the dataset. Count word frequencies, create a mapping word->int"""
    self.wordcount = collections.Counter(self._words_in_file())
    idx = 0
    self.dict = {}
    # Assign a number to every word we have. 0 = the most common word
    for word, _ in self.wordcount.most_common():
      # We only want vocab_size words in or dictionary. Skip the remaining uncommon words
      if idx == self.vocab_size:
        break
      self.dict[word] = idx
      idx += 1
    self.reversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

  def _words_in_file(self):
    """Returns a generator over all words in the file"""
    with open(self.filename) as f:
      for line in f:
        words = line.split()
        for word in words:
          yield word

  def string_samples(self):
    """ Returns a generator for samples (targetword->contextword) """
    # possible positions of context-words relative to a target word
    contextoffsets = [x for x in range(-self.skip_window,self.skip_window+1) if x != 0]
    with open(self.filename) as f:
      for line in f:
        idx = 0
        words = line.split()
        for word in words:
          # choose a random context word. Take special care to stay in the bounds of the list
          contextoffset = random.choice(contextoffsets)
          contextindex = idx+contextoffset
          # if selected index-offset reaches outside of the list, try the other direction
          if idx+contextoffset < 0 or idx+contextoffset >= len(words):
            contextoffset = contextoffset*-1
            # above fails if the current line is to short. Stay inside the bounds at all cost!
            contextindex = min(len(words)-1,max(0,idx+contextoffset))
          yield (word,words[contextindex])
          idx += 1

  def _lookup_label(self,gen):
    """ Maps the words in the input-tuple to numbers"""
    for e in gen:
      try:
        yield (self.dict[e[0]],self.dict[e[1]])
      except KeyError:
        pass

  def _repeat(self,generator_func):
    """ Repeat a given generator forever """
    g = generator_func()
    while True:
      try:
        yield g.__next__()
      except StopIteration:
        g = generator_func()

  def _batch(self,samples):
    """ Pack self.batch_size of training samples into a batch 
        The output is a tuple of two lists, rather then a list of tuples, because this way we can treat
        the two lists as input-tensor and label-tensor.
        The second list is al list of one-element-lists, because tf.nce_loss wants its tensor in that shape"""
    while True:
      inputs = []
      labels = []
      for _ in range(0,self.batch_size):
        sample = samples.__next__()
        inputs.append(sample[0])
        labels.append([sample[1]])
      yield inputs,labels

  def batches(self):
    """ Returns a generator the will yield an infinite amout of training-batches ready to feed into the model"""
    return self._batch(self._lookup_label(self._repeat(self.string_samples)))