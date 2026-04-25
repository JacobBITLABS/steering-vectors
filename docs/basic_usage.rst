Basic usage
===========
This library assumes you're using PyTorch with a decoder-only generative language model (e.g. GPT, LLaMa, etc...), and a tokenizer from Huggingface.

To begin, collect tuples of positive and negative training prompts in a list, and run ``train_steering_vector()``:

.. code-block:: python

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from steering_vectors import train_steering_vector

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # training samples are tuples of (positive_prompt, negative_prompt)
    training_samples = [
        (
            "The capital of England is London",
            "The capital of England is Beijing"
        ),
        (
            "The capital of France is Paris",
            "The capital of France is Berlin"
        )
        # ...
    ]

    steering_vector = train_steering_vector(
        model,
        tokenizer,
        training_samples,
        show_progress=True,
    )

Then, you can use the steering vector to "steer" the model's behavior, for example to make it more truthful:

.. code-block:: python

    with steering_vector.apply(model):
        prompt = "Is it true that crystals have magic healing properties?"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs)


Using specific layers
'''''''''''''''''''''

By default, ``train_steering_vector()`` will train a vector at all layers of the model.
However, often you may want to only train a steering vector for a limited set of layers.
You can customize the layers to train by passing a list of layer numbers to the ``layers`` argument:

.. code-block:: python

    steering_vec = train_steering_vector(
        model,
        tokenizer,
        training_samples,
        layers=[1, 2, 3],
    )

This also works with negative indices to count from the end of the model:

.. code-block:: python

    steering_vec = train_steering_vector(
        model,
        tokenizer,
        training_samples,
        layers=[-1, -2, -3],
    )


Batch training
''''''''''''''

By default, ``train_steering_vector()`` will use a batch size of 1. If your
GPU has enough memory, you can increase the batch size to train faster by
setting the ``batch_size`` argument:

.. code-block:: python

    steering_vec = train_steering_vector(
        model,
        tokenizer,
        training_samples,
        batch_size=8,
    )

When using the default mean aggregator, training is streamed batch-by-batch, so
the library does not keep every activation tensor for the whole dataset in GPU
memory. Increasing ``batch_size`` still increases peak activation memory for
each forward pass.

If you want to override that behavior, pass ``activation_mode`` to
``train_steering_vector()``. Use ``"stream"`` to force the running-mean path,
``"materialize"`` to keep the legacy full-activation path, or ``"auto"``
to let the library choose based on the aggregator.


Magnitude scaling
'''''''''''''''''

By default, the steering vector will be applied at full magnitude. However, sometimes it's useful
to apply the steering vector at a lower or higher magnitude, depending on the application. This
can be done by passing a ``multiplier`` argument to ``steering_vector.apply()`` or ``steering_vector.patch()``:

.. code-block:: python

    with steering_vec.apply(model, multiplier=0.5):
        # the steering vector will be applied at half magnitude
        model.forward(...)

    with steering_vec.apply(model, multiplier=2.0):
        # the steering vector will be applied at double magnitude
        model.forward(...)
    
    with steering_vec.apply(model, multiplier=-1.0):
        # the steering vector will be inverted
        model.forward(...)
