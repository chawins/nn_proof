from abc import ABCMeta
import collections
import warnings
import numpy as np
from six.moves import xrange
import tensorflow as tf

# from cleverhans.attacks import Attack
from cleverhans.attacks_tf import _logger, np_dtype, tf_dtype, ZERO
from cleverhans.compat import (reduce_any, reduce_max, reduce_mean, reduce_min,
                               reduce_sum)
from cleverhans import utils
from cleverhans import utils_tf


class CarliniWagnerL2(object):
    """
    This attack is adapted from Cleverhans implementation of Carlini-Wagner L2
    attack. It attacks the main model along with an ensemble of high-level
    feature detectors as a defense, assuming that gradient and output of all
    models are accessible to the adversary.

    To maintain a similar structure to Cleverhans, it is used as a wrapper of
    CarliniWagnerL2_TF_WB.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, back='tf', sess=None, dtypestr='float32'):
        """
        All params are the same as the original version in Cleverhans. There is
        one extra param:

        :param ensemble: (required) a list (indexed by class) of lists of
                         high-level feature detectors (order does not matter for
                         the inner listss)

        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        # To allow custom model, copy init from Attack object
        # if not isinstance(model, Model):
        #     model = CallableModelWrapper(model, 'logits')

        # super(CarliniWagnerL2, self).__init__(model, back, sess, dtypestr)
        import tensorflow as tf
        self.tf_dtype = tf.as_dtype(dtypestr)
        self.np_dtype = np.dtype(dtypestr)

        if sess is None:
            sess = tf.get_default_session()
        if not isinstance(sess, tf.Session):
            raise TypeError("sess is not an instance of tf.Session")

        from cleverhans import attacks_tf
        attacks_tf.np_dtype = self.np_dtype
        attacks_tf.tf_dtype = self.tf_dtype

        self.model = model
        self.sess = sess
        self.dtypestr = dtypestr
        self.graphs = {}
        self.feedable_kwargs = {}
        self.structural_kwargs = []

        import tensorflow as tf
        self.feedable_kwargs = {'y': self.tf_dtype, 'y_target': self.tf_dtype}

        self.structural_kwargs = [
            'batch_size', 'confidence', 'targeted', 'learning_rate',
            'binary_search_steps', 'max_iterations', 'abort_early',
            'initial_const', 'clip_min', 'clip_max'
        ]

    def generate(self, x, **kwargs):
        """
        Return a tensor that constructs adversarial examples for the given
        input. Generate uses tf.py_func in order to operate over tensors.
        :param x: (required) A tensor with the inputs.
        :param y: (optional) A tensor with the true labels for an untargeted
                  attack. If None (and y_target is None) then use the
                  original labels the classifier assigns.
        :param y_target: (optional) A tensor with the target labels for a
                  targeted attack.
        :param confidence: Confidence of adversarial examples: higher produces
                           examples with larger l2 distortion, but more
                           strongly classified as adversarial.
        :param batch_size: Number of attacks to run simultaneously.
        :param learning_rate: The learning rate for the attack algorithm.
                              Smaller values produce better results but are
                              slower to converge.
        :param binary_search_steps: The number of times we perform binary
                                    search to find the optimal tradeoff-
                                    constant between norm of the purturbation
                                    and confidence of the classification.
        :param max_iterations: The maximum number of iterations. Setting this
                               to a larger value will produce lower distortion
                               results. Using only a few iterations requires
                               a larger learning rate, and will produce larger
                               distortion results.
        :param abort_early: If true, allows early aborts if gradient descent
                            is unable to make progress (i.e., gets stuck in
                            a local minimum).
        :param initial_const: The initial tradeoff-constant to use to tune the
                              relative importance of size of the pururbation
                              and confidence of classification.
                              If binary_search_steps is large, the initial
                              constant is not important. A smaller value of
                              this constant gives lower distortion results.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        import tensorflow as tf
        self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)

        attack = CarliniWagnerL2_TF(
            self.sess, self.model, self.batch_size, 
            self.confidence, 'y_target' in kwargs, self.learning_rate,
            self.binary_search_steps, self.max_iterations, self.abort_early,
            self.initial_const, self.clip_min, self.clip_max, nb_classes,
            x.get_shape().as_list()[1:])

        def cw_wrap(x_val, y_val):
            return np.array(attack.attack(x_val, y_val), dtype=self.np_dtype)

        wrap = tf.py_func(cw_wrap, [x, labels], self.tf_dtype)
        wrap.set_shape(x.get_shape())

        return wrap

    def parse_params(self,
                     y=None,
                     y_target=None,
                     nb_classes=None,
                     batch_size=1,
                     confidence=0,
                     learning_rate=5e-3,
                     binary_search_steps=5,
                     max_iterations=1000,
                     abort_early=True,
                     initial_const=1e-2,
                     clip_min=0,
                     clip_max=1):

        # ignore the y and y_target argument
        if nb_classes is not None:
            warnings.warn("The nb_classes argument is depricated and will "
                          "be removed on 2018-02-11")
        self.batch_size = batch_size
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.clip_min = clip_min
        self.clip_max = clip_max

    def construct_graph(self, fixed, feedable, x_val, hash_key):
        """
        Construct the graph required to run the attack through generate_np.
        :param fixed: Structural elements that require defining a new graph.
        :param feedable: Arguments that can be fed to the same graph when
                        they take different values.
        :param x_val: symbolic adversarial example
        :param hash_key: the key used to store this graph in our cache
        """
        # try our very best to create a TF placeholder for each of the
        # feedable keyword arguments, and check the types are one of
        # the allowed types
        class_name = str(self.__class__).split(".")[-1][:-2]
        _logger.info("Constructing new graph for attack " + class_name)

        # remove the None arguments, they are just left blank
        for k in list(feedable.keys()):
            if feedable[k] is None:
                del feedable[k]

        # process all of the rest and create placeholders for them
        new_kwargs = dict(x for x in fixed.items())
        for name, value in feedable.items():
            given_type = self.feedable_kwargs[name]
            if isinstance(value, np.ndarray):
                new_shape = [None] + list(value.shape[1:])
                new_kwargs[name] = tf.placeholder(given_type, new_shape)
            elif isinstance(value, utils.known_number_types):
                new_kwargs[name] = tf.placeholder(given_type, shape=[])
            else:
                raise ValueError("Could not identify type of argument " +
                                 name + ": " + str(value))

        # x is a special placeholder we always want to have
        x_shape = [None] + list(x_val.shape)[1:]
        x = tf.placeholder(self.tf_dtype, shape=x_shape)

        # now we generate the graph that we want
        x_adv = self.generate(x, **new_kwargs)

        self.graphs[hash_key] = (x, new_kwargs, x_adv)

        if len(self.graphs) >= 10:
            warnings.warn("Calling generate_np() with multiple different "
                          "structural paramaters is inefficient and should"
                          " be avoided. Calling generate() is preferred.")

    def generate_np(self, x_val, **kwargs):
        """
        Generate adversarial examples and return them as a NumPy array.
        Sub-classes *should not* implement this method unless they must
        perform special handling of arguments.
        :param x_val: A NumPy array with the original inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A NumPy array holding the adversarial examples.
        """
        if self.sess is None:
            raise ValueError("Cannot use `generate_np` when no `sess` was"
                             " provided")

        fixed, feedable, hash_key = self.construct_variables(kwargs)

        if hash_key not in self.graphs:
            self.construct_graph(fixed, feedable, x_val, hash_key)
        else:
            # remove the None arguments, they are just left blank
            for k in list(feedable.keys()):
                if feedable[k] is None:
                    del feedable[k]

        x, new_kwargs, x_adv = self.graphs[hash_key]

        feed_dict = {x: x_val}

        for name in feedable:
            feed_dict[new_kwargs[name]] = feedable[name]

        return self.sess.run(x_adv, feed_dict)

    def construct_variables(self, kwargs):
        """
        Construct the inputs to the attack graph to be used by generate_np.
        :param kwargs: Keyword arguments to generate_np.
        :return: Structural and feedable arguments as well as a unique key
                for the graph given these inputs.
        """
        # the set of arguments that are structural properties of the attack
        # if these arguments are different, we must construct a new graph
        fixed = dict(
            (k, v) for k, v in kwargs.items() if k in self.structural_kwargs)

        # the set of arguments that are passed as placeholders to the graph
        # on each call, and can change without constructing a new graph
        feedable = dict(
            (k, v) for k, v in kwargs.items() if k in self.feedable_kwargs)

        if len(fixed) + len(feedable) < len(kwargs):
            warnings.warn("Supplied extra keyword arguments that are not "
                          "used in the graph computation. They have been "
                          "ignored.")

        if not all(isinstance(value, collections.Hashable) 
            for value in fixed.values()):
            # we have received a fixed value that isn't hashable
            # this means we can't cache this graph for later use,
            # and it will have to be discarded later
            hash_key = None
        else:
            # create a unique key for this set of fixed paramaters
            hash_key = tuple(sorted(fixed.items()))

        return fixed, feedable, hash_key

    def get_or_guess_labels(self, x, kwargs):
        """
        Get the label to use in generating an adversarial example for x.
        The kwargs are fed directly from the kwargs of the attack.
        If 'y' is in kwargs, then assume it's an untargeted attack and
        use that as the label.
        If 'y_target' is in kwargs and is not none, then assume it's a
        targeted attack and use that as the label.
        Otherwise, use the model's prediction as the label and perform an
        untargeted attack.
        """
        if 'y' in kwargs and 'y_target' in kwargs:
            raise ValueError("Can not set both 'y' and 'y_target'.")
        elif 'y' in kwargs:
            labels = kwargs['y']
        elif 'y_target' in kwargs and kwargs['y_target'] is not None:
            labels = kwargs['y_target']
        else:
            preds = self.model.get_output(x)
            preds_max = reduce_max(preds, 1, keepdims=True)
            original_predictions = tf.to_float(tf.equal(preds, preds_max))
            labels = tf.stop_gradient(original_predictions)
        if isinstance(labels, np.ndarray):
            nb_classes = labels.shape[1]
        else:
            nb_classes = labels.get_shape().as_list()[1]
        return labels, nb_classes


class CarliniWagnerL2_TF(object):

    def __init__(self, sess, model, batch_size, confidence, targeted,
                 learning_rate, binary_search_steps, max_iterations,
                 abort_early, initial_const, clip_min, clip_max, num_labels,
                 shape):
        """
        """

        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.repeat = binary_search_steps >= 10

        self.shape = shape = tuple([batch_size] + list(shape))

        # the variable we're going to optimize over
        modifier = tf.Variable(np.zeros(shape, dtype=np_dtype))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf_dtype, name='timg')
        self.tlab = tf.Variable(
            np.zeros((batch_size, num_labels)), dtype=tf_dtype, name='tlab')
        self.const = tf.Variable(
            np.zeros(batch_size), dtype=tf_dtype, name='const')

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf_dtype, shape, name='assign_timg')
        self.assign_tlab = tf.placeholder(
            tf_dtype, (batch_size, num_labels), name='assign_tlab')
        self.assign_const = tf.placeholder(
            tf_dtype, [batch_size], name='assign_const')

        # the resulting instance, tanh'd to keep bounded from clip_min
        # to clip_max
        self.newimg = (tf.tanh(modifier + self.timg) + 1) / 2
        self.newimg = self.newimg * (clip_max - clip_min) + clip_min

        # distance to the input data
        other = (tf.tanh(self.timg) + 1) / \
            2 * (clip_max - clip_min) + clip_min
        self.l2dist = reduce_sum(
            tf.square(self.newimg - other), list(range(1, len(shape))))

        # prediction BEFORE-SOFTMAX of the model
        output = model.get_output(self.newimg)
        self.output = output
        # compute the probability of the label class versus the maximum other
        real = reduce_sum((self.tlab) * output, 1)
        other = reduce_max((1 - self.tlab) * output - self.tlab * 10000, 1)
        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

        # Sum up the losses
        self.loss1 = self.const * loss1
        self.loss = reduce_mean(self.loss1 + self.l2dist)

        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given instance for the given targets.
        If self.targeted is true, then the targets represents the target labels
        If self.targeted is false, then targets are the original class labels
        """

        r = []
        for i in range(0, len(imgs), self.batch_size):
            _logger.debug(
                ("Running CWL2 attack on instance " + "{} of {}").format(
                    i, len(imgs)))
            r.extend(
                self.attack_batch(imgs[i:i + self.batch_size],
                                  targets[i:i + self.batch_size]))
        return np.array(r)

    def attack_batch(self, imgs, labs):
        """
        Run the attack on a batch of instance and labels.
        """

        def compare(x, y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        oimgs = np.clip(imgs, self.clip_min, self.clip_max)

        # re-scale instances to be within range [0, 1]
        imgs = (imgs - self.clip_min) / (self.clip_max - self.clip_min)
        imgs = np.clip(imgs, 0, 1)
        # now convert to [-1, 1]
        imgs = (imgs * 2) - 1
        # convert to tanh-space
        imgs = np.arctanh(imgs * .999999)

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # placeholders for the best l2, score, and instance attack found so far
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(oimgs)

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            _logger.debug("  Binary search step {} of {}".format(
                outer_step, self.BINARY_SEARCH_STEPS))

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
                CONST = upper_bound

            # set the variables so that we don't have to send them over again
            self.sess.run(
                self.setup, {
                    self.assign_timg: batch,
                    self.assign_tlab: batchlab,
                    self.assign_const: CONST
                })

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg = self.sess.run([
                    self.train, self.loss, self.l2dist, self.output,
                    self.newimg])

                if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                                   "l2={:.3g}").format(
                                       iteration, self.MAX_ITERATIONS, l,
                                       np.mean(l2s)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and \
                   iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if l > prev * .9999:
                        msg = "    Failed to make progress; stop early"
                        _logger.debug(msg)
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, ii) in enumerate(zip(l2s, scores, nimg)):
                    lab = np.argmax(batchlab[e])
                    if l2 < bestl2[e] and compare(sc, lab):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and compare(sc, lab):
                        o_bestl2[e] = l2
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and \
                   bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        CONST[e] *= 10
            _logger.debug("  Successfully generated adversarial examples " +
                          "on {} of {} instances.".format(
                              sum(upper_bound < 1e9), batch_size))
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(np.sqrt(o_bestl2[o_bestl2 < 1e9]))
            _logger.debug("   Mean successful distortion: {:.4g}".format(mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack
