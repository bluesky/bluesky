Scan a grid around each sample in a grid
----------------------------------------

.. literalinclude:: grid_in_grid.py

Demo output:

.. plot:: cookbook/grid_in_grid.py

.. ipython:: python
    :suppress:

    from bluesky import RunEngine
    RE = RunEngine({})
    %run -i source/cookbook/grid_in_grid.py

.. ipython:: python

    RE(grid_in_grid(samples))
