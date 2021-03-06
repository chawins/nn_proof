import warnings

from cleverhans.attacks import Attack
from cleverhans.attacks_tf import _logger, np_dtype, tf_dtype, ZERO
from cleverhans.compat import (reduce_any, reduce_max, reduce_mean, reduce_min,
                               reduce_sum)
from cleverhans.model import CallableModelWrapper, Model
from parameters import *


class CarliniWagnerL2_WB(Attack):
    """
    This attack is adapted from Cleverhans implementation of Carlini-Wagner L2
    attack. It attacks the main model along with an ensemble of high-level
    feature detectors as a defense, assuming that gradient and output of all
    models are accessible to the adversary.

    To maintain a similar structure to Cleverhans, it is used as a wrapper of
    CarliniWagnerL2_TF_WB.
    """

    def __init__(self, model, ensemble=None, back='tf', sess=None,
                 dtypestr='float32'):
        """
        All params are the same as the original version in Cleverhans. There is
        one extra param:

        :param ensemble: (required) a list (indexed by class) of lists of
                         high-level feature detectors (order does not matter for
                         the inner listss)

        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'logits')

        super(CarliniWagnerL2_WB, self).__init__(model, back, sess, dtypestr)
        self.ensemble = ensemble

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

        attack = CarliniWagnerL2_WB_TF(
            self.sess, self.model, self.ensemble, self.batch_size,
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


class CarliniWagnerL2_WB_TF(object):

    def __init__(self, sess, model, ensemble, batch_size, confidence, targeted,
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
        self.model = model
        self.ensemble = ensemble

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

        # prediction BEFORE-SOFTMAX of the model
        self.output = model.get_logits(self.newimg)

        # distance to the input data
        self.other = (tf.tanh(self.timg) + 1) / \
            2 * (clip_max - clip_min) + clip_min
        self.l2dist = reduce_sum(
            tf.square(self.newimg - self.other), list(range(1, len(shape))))

        # compute the probability of the label class versus the maximum other
        real = reduce_sum((self.tlab) * self.output, 1)
        other = reduce_max((1 - self.tlab) * self.output -
                           self.tlab * 10000, 1)

        if self.TARGETED:
            # if targeted, optimize for making the other class most likely
            loss1 = tf.maximum(ZERO(), other - real + self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(ZERO(), real - other + self.CONFIDENCE)

        # sum up the losses
        self.loss2 = reduce_sum(self.l2dist)

        # ==================== Add ensemble part ==================== #
        # Get the number of small nets for each class
        self.n_nets = np.array([len(x) for x in self.ensemble])
        # Max number of small nets in one class
        n_nets_max = np.max(self.n_nets)

        # Gather all outputs from the ensemble
        all_nets = []
        for i in range(num_labels):
            class_nets = []
            for j in range(n_nets_max):
                if j < self.n_nets[i]:
                    class_nets.append(
                        self.ensemble[i][j].get_logits(self.newimg))
                else:
                    # Padding: append [0, 0] for classes that have the number
                    # of NNs less than n_nets_max
                    class_nets.append(tf.zeros([batch_size, 2]))
            all_nets.append(tf.stack(class_nets, axis=1))
        self.ensemble_out = tf.stack(all_nets, axis=1)

        # Based on output, see which set of the ensemble to use
        # Find label/class to look for in ensemble
        if self.TARGETED:
            label = tf.argmax(self.tlab, axis=1)
        else:
            # Output of original image
            self.orig_output = model.get_logits(self.other)
            label = tf.argmax(self.orig_output, axis=1)
        ind = tf.range(batch_size, dtype=tf.int64)
        ind_label = tf.stack([ind, label], axis=1)
        # Use gather_nd to do numpy slicing
        self.label_nets = tf.gather_nd(self.ensemble_out, ind_label)

        # DEBUG
        # print("self.ensemble_out: ", self.ensemble_out)
        # print("label: ", label)
        # print("ind_label: ", ind_label)
        # print("label_nets: ", self.label_nets)

        # Get the loss function for the small net part
        if self.TARGETED:
            diff = self.label_nets[:, :, 0] - self.label_nets[:, :, 1]
        else:
            diff = self.label_nets[:, :, 1] - self.label_nets[:, :, 0]
        # Find the largest difference among small nets
        max_diff = tf.reduce_max(diff, axis=1)
        # Add confidence margin and clip at zero
        ensemble_loss = tf.maximum(ZERO(), max_diff + self.CONFIDENCE)
        # The objective function only includes max(clf_loss, any_ensemble_loss)
        loss1 = tf.maximum(loss1, tf.squeeze(ensemble_loss))
        self.loss1 = reduce_sum(self.const * loss1)
        self.loss = self.loss1 + self.loss2

        # DEBUG
        # print("max_diff: ", max_diff)
        # print("ensemble_loss: ", ensemble_loss)
        # print("loss1: ", loss1)
        # print("reduce_sum loss1: ", self.loss1)

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

        def check_ens(sc, sce, lab):
            """
            Check if <img> succeed. Change rule later.
            :param sce: output of ensemble chosen at the <lab> label
            """
            sce_lab = np.copy(sce)
            n_net_lab = self.n_nets[lab]
            if not isinstance(sc, (float, int, np.int64)):
                sc = np.copy(sc)
                if self.TARGETED:
                    sc[lab] -= self.CONFIDENCE
                    sce_lab = np.array([[net[0], net[1] - self.CONFIDENCE]
                                        for net in sce_lab[:n_net_lab]])
                else:
                    sc[lab] += self.CONFIDENCE
                    sce_lab = np.array([[net[0] + self.CONFIDENCE, net[1]]
                                        for net in sce_lab[:n_net_lab]])
                sc = np.argmax(sc)
                n_net_suc = np.sum(np.argmax(sce_lab, axis=1))
            if self.TARGETED:
                return (sc == lab) and (n_net_suc == n_net_lab)
            else:
                return (sc != lab) and (n_net_suc == 0)

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

            # DEBUG
            # ensemble_out = self.sess.run([self.ensemble_out])
            # _logger.debug(ensemble_out)

            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack
                _, l, l2s, scores, nimg, scores_ens = self.sess.run([
                    self.train, self.loss, self.l2dist, self.output,
                    self.newimg, self.label_nets
                ])

                if iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    _logger.debug(("    Iteration {} of {}: loss={:.3g} " +
                                   "l2={:.3g} f={:.3g}").format(
                                       iteration, self.MAX_ITERATIONS, l,
                                       np.mean(l2s), np.mean(scores)))

                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and \
                   iteration % ((self.MAX_ITERATIONS // 10) or 1) == 0:
                    if l > prev * .9999:
                        msg = "    Failed to make progress; stop early"
                        _logger.debug(msg)
                        break
                    prev = l

                # adjust the best result found so far
                for e, (l2, sc, sce, ii) in enumerate(zip(l2s, scores,
                                                          scores_ens, nimg)):
                    lab = np.argmax(batchlab[e])
                    if l2 < bestl2[e] and check_ens(sc, sce, lab):
                        bestl2[e] = l2
                        bestscore[e] = np.argmax(sc)
                    if l2 < o_bestl2[e] and check_ens(sc, sce, lab):
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
