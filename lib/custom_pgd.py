from abc import ABCMeta
import collections
import warnings
import numpy as np
from six.moves import xrange
import tensorflow as tf

# from cleverhans.attacks import Attack
from cleverhans.attacks_tf import _logger, np_dtype, tf_dtype, ZERO
from cleverhans.compat import (reduce_any, reduce_max, reduce_mean, reduce_min,
                               reduce_sum, softmax_cross_entropy_with_logits)
from cleverhans import utils
from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta

CONF = 1e-2

class FastGradientMethod(object):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the
    infinity norm (and is known as the "Fast Gradient Sign Method"). This
    implementation extends the attack to other norms, and is therefore called
    the Fast Gradient Method.
    Paper link: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model, model2, thres, sess=None, dtypestr='float32', **kwargs):
        """
        Create a FastGradientMethod instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        # super(FastGradientMethod, self).__init__(model, sess, dtypestr, **kwargs)
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
        self.model2 = model2
        self.thres = thres
        self.sess = sess
        self.dtypestr = dtypestr
        self.graphs = {}
        self.feedable_kwargs = {}
        self.structural_kwargs = []

        self.feedable_kwargs = {
            'eps': self.np_dtype,
            'y': self.np_dtype,
            'y_target': self.np_dtype,
            'clip_min': self.np_dtype,
            'clip_max': self.np_dtype
        }
        self.structural_kwargs = ['ord']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                this parameter if you'd like to use true labels when crafting
                adversarial samples. Otherwise, model predictions are used as
                labels to avoid the "label leaking" effect (explained in this
                paper: https://arxiv.org/abs/1611.01236). Default is None.
                Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                        y_target=None if y is also set. Labels should be
                        one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, _nb_classes = self.get_or_guess_labels(x, kwargs)

        return fgm(x,
                self.model.get_logits(x),
                self.model2.get_output(x),
                self.thres,
                y=labels,
                eps=self.eps,
                ord=self.ord,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                targeted=(self.y_target is not None))

    def parse_params(self,
                    eps=0.3,
                    ord=np.inf,
                    y=None,
                    y_target=None,
                    clip_min=None,
                    clip_max=None,
                    **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics NumPy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A tensor with the model labels. Only provide
                this parameter if you'd like to use true labels when crafting
                adversarial samples. Otherwise, model predictions are used as
                labels to avoid the "label leaking" effect (explained in this
                paper: https://arxiv.org/abs/1611.01236). Default is None.
                Labels should be one-hot-encoded.
        :param y_target: (optional) A tensor with the labels to target. Leave
                        y_target=None if y is also set. Labels should be
                        one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters

        self.eps = eps
        self.ord = ord
        self.y = y
        self.y_target = y_target
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        return True

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
            preds = self.model.get_probs(x)
            preds_max = reduce_max(preds, 1, keepdims=True)
            original_predictions = tf.to_float(tf.equal(preds, preds_max))
            labels = tf.stop_gradient(original_predictions)
        if isinstance(labels, np.ndarray):
            nb_classes = labels.shape[1]
        else:
            nb_classes = labels.get_shape().as_list()[1]
        return labels, nb_classes

def fgm(x,
        logits,
        model2_output,
        thres,
        y=None,
        eps=0.3,
        ord=np.inf,
        clip_min=None,
        clip_max=None,
        targeted=False):
    """
    TensorFlow implementation of the Fast Gradient Method.
    :param x: the input placeholder
    :param logits: output of model.get_logits
    :param y: (optional) A placeholder for the model labels. If targeted
                is true, then provide the target label. Otherwise, only provide
                this parameter if you'd like to use true labels when crafting
                adversarial samples. Otherwise, model predictions are used as
                labels to avoid the "label leaking" effect (explained in this
                paper: https://arxiv.org/abs/1611.01236). Default is None.
                Labels should be one-hot-encoded.
    :param eps: the epsilon (input variation parameter)
    :param ord: (optional) Order of the norm (mimics NumPy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: Minimum float value for adversarial example components
    :param clip_max: Maximum float value for adversarial example components
    :param targeted: Is the attack targeted or untargeted? Untargeted, the
                    default, will try to make the label incorrect. Targeted
                    will instead try to move in the direction of being more
                    like y.
    :return: a tensor for the adversarial example
    """

    # Make sure the caller has not passed probs by accident
    assert logits.op.type != 'Softmax'

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = reduce_max(logits, 1, keepdims=True)
        y = tf.to_float(tf.equal(logits, preds_max))
        y = tf.stop_gradient(y)
    y = y / reduce_sum(y, 1, keepdims=True)

    # Compute loss
    # loss = softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # if targeted:
    #     loss = -loss
    # Hinge loss
    real = reduce_sum((y) * logits, 1)
    other = reduce_max((1 - y) * logits - y * 1e9, 1)
    if targeted:
        # if targeted, optimize for making the other class most likely
        loss = tf.maximum(ZERO(), other - real + CONF)
    else:
        # if untargeted, optimize for making this class least likely.
        loss = tf.maximum(ZERO(), real - other + CONF)
    loss = -loss

    # Add model2
    # For output after sigmoid
    # loss2 = tf.maximum(ZERO(), thres + CONF - model2_output)
    # For output before sigmoid (this should get the total output to ~2, but
    # surprisingly, this mostly gets the total output above 3)
    loss2 = tf.reduce_sum(tf.maximum(ZERO(), CONF - model2_output), axis=1)
    # For custom_activation [-1, 1]
    # loss2 = tf.reduce_sum(tf.maximum(ZERO(), CONF - model2_output), axis=1)
    loss = loss - loss2
    # loss = -loss2

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    if ord == np.inf:
        # Take sign of gradient
        normalized_grad = tf.sign(grad)
        # The following line should not change the numerical results.
        # It applies only because `normalized_grad` is the output of
        # a `sign` op, which has zero derivative anyway.
        # It should not be applied for the other norms, where the
        # perturbation has a non-zero derivative.
        normalized_grad = tf.stop_gradient(normalized_grad)
    elif ord == 1:
        red_ind = list(xrange(1, len(x.get_shape())))
        avoid_zero_div = 1e-12
        avoid_nan_norm = tf.maximum(avoid_zero_div,
                                    reduce_sum(tf.abs(grad),
                                            reduction_indices=red_ind,
                                            keepdims=True))
        normalized_grad = grad / avoid_nan_norm
    elif ord == 2:
        red_ind = list(xrange(1, len(x.get_shape())))
        avoid_zero_div = 1e-12
        square = tf.maximum(avoid_zero_div,
                            reduce_sum(tf.square(grad),
                                    reduction_indices=red_ind,
                                    keepdims=True))
        normalized_grad = grad / tf.sqrt(square)
    else:
        raise NotImplementedError("Only L-inf, L1 and L2 norms are "
                                "currently implemented.")

    # Multiply by constant epsilon
    scaled_grad = eps * normalized_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + scaled_grad

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) and (clip_max is not None):
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    return adv_x


class CustomPGD(object):
    """
    This class implements either the Basic Iterative Method
    (Kurakin et al. 2016) when rand_init is set to 0. or the
    Madry et al. (2017) method when rand_minmax is larger than 0.
    Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
    Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
    """
    __metaclass__ = ABCMeta

    FGM_CLASS = FastGradientMethod

    def __init__(self, model, model2, thres, sess=None, dtypestr='float32',
                 default_rand_init=True, **kwargs):
        """
        Create a ProjectedGradientDescent instance.
        Note: the model parameter should be an instance of the
        cleverhans.model.Model abstraction provided by CleverHans.
        """

        # super(ProjectedGradientDescent, self).__init__(model, sess=sess,
        #                                                 dtypestr=dtypestr, **kwargs)
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
        self.model2 = model2
        self.thres = thres
        self.sess = sess
        self.dtypestr = dtypestr
        self.graphs = {}
        self.feedable_kwargs = {}
        self.structural_kwargs = []

        self.feedable_kwargs = {
            'eps': self.np_dtype,
            'eps_iter': self.np_dtype,
            'y': self.np_dtype,
            'y_target': self.np_dtype,
            'clip_min': self.np_dtype,
            'clip_max': self.np_dtype
        }
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init', 'sanity_checks']
        self.default_rand_init = default_rand_init

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param eps: (optional float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (optional float) step size for each attack iteration
        :param nb_iter: (optional int) Number of attack iterations.
        :param rand_init: (optional) Whether to use random initialization
        :param y: (optional) A tensor with the true class labels
            NOTE: do not use smoothed labels here
        :param y_target: (optional) A tensor with the labels to target. Leave
                            y_target=None if y is also set. Labels should be
                            one-hot-encoded.
            NOTE: do not use smoothed labels here
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Initialize loop variables
        if self.rand_init:
            eta = tf.random_uniform(tf.shape(x), -self.rand_minmax,
                                    self.rand_minmax, dtype=self.tf_dtype)
        else:
            eta = tf.zeros(tf.shape(x))
        eta = clip_eta(eta, self.ord, self.eps)

        # Fix labels to the first model predictions for loss computation
        model_preds = self.model.get_logits(x)
        preds_max = reduce_max(model_preds, 1, keepdims=True)
        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            y = tf.to_float(tf.equal(model_preds, preds_max))
            y = tf.stop_gradient(y)
            targeted = False

        y_kwarg = 'y_target' if targeted else 'y'
        fgm_params = {
            'eps': self.eps_iter,
            y_kwarg: y,
            'ord': self.ord,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max
        }

        # Use getattr() to avoid errors in eager execution attacks
        FGM = self.FGM_CLASS(
            self.model,
            self.model2,
            self.thres,
            sess=getattr(self, 'sess', None),
            dtypestr=self.dtypestr)

        def cond(i, _):
            return tf.less(i, self.nb_iter)

        def body(i, e):
            adv_x = FGM.generate(x + e, **fgm_params)

            # Clipping perturbation according to clip_min and clip_max
            if self.clip_min is not None and self.clip_max is not None:
                adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

            # Clipping perturbation eta to self.ord norm ball
            eta = adv_x - x
            eta = clip_eta(eta, self.ord, self.eps)
            return i + 1, eta

        _, eta = tf.while_loop(cond, body, [tf.zeros([]), eta], back_prop=True)

        # Define adversarial example (and clip if necessary)
        adv_x = x + eta
        if self.clip_min is not None or self.clip_max is not None:
            assert self.clip_min is not None and self.clip_max is not None
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        asserts = []

        # Asserts run only on CPU.
        # When multi-GPU eval code tries to force all PGD ops onto GPU, this
        # can cause an error.
        with tf.device("/CPU:0"):
            asserts.append(tf.assert_less_equal(self.eps_iter, self.eps))
            if self.ord == np.inf and self.clip_min is not None:
            # The 1e-6 is needed to compensate for numerical error.
            # Without the 1e-6 this fails when e.g. eps=.2, clip_min=.5, clip_max=.7
                asserts.append(tf.assert_less_equal(
                    self.eps, 1e-6 + self.clip_max - self.clip_min))

        if self.sanity_checks:
            with tf.control_dependencies(asserts):
                adv_x = tf.identity(adv_x)

        return adv_x

    def parse_params(self,
                    eps=0.3,
                    eps_iter=0.05,
                    nb_iter=10,
                    y=None,
                    ord=np.inf,
                    clip_min=None,
                    clip_max=None,
                    y_target=None,
                    rand_init=None,
                    rand_minmax=0.3,
                    sanity_checks=True,
                    **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (optional float) step size for each attack iteration
        :param nb_iter: (optional int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                            y_target=None if y is also set. Labels should be
                            one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param sanity_checks: bool Insert tf asserts checking values
            (Some tests need to run with no sanity checks because the
                tests intentionally configure the attack strangely)
        """

        # Save attack-specific parameters
        self.eps = eps
        if rand_init is None:
            rand_init = self.default_rand_init
        self.rand_init = rand_init
        if self.rand_init:
            self.rand_minmax = eps
        else:
            self.rand_minmax = 0.
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        self.sanity_checks = sanity_checks

        return True

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
            preds = self.model.get_probs(x)
            preds_max = reduce_max(preds, 1, keepdims=True)
            original_predictions = tf.to_float(tf.equal(preds, preds_max))
            labels = tf.stop_gradient(original_predictions)
        if isinstance(labels, np.ndarray):
            nb_classes = labels.shape[1]
        else:
            nb_classes = labels.get_shape().as_list()[1]
        return labels, nb_classes

