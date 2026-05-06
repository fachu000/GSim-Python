
- The single most important rule is: our code should produce correct results or raise an error. 

- Don't repeat yourself (DRY). Reuse existing methods whenever possible. Extend their functionality if needed. Avoid having methods with overlapping functionality; it is preferable to factor out the common functionality into a single method that can be reused by other methods.

- Please ensure that each function or class is defined in a location that is consistent with the scope of its functionality. Examples:
        
    - If a method assumes that the input data has the format of a specific class or module (e.g. a matrix of location pairs that is M x 6), it should be defined within that class or module (e.g. CgMeas).

    - If a function is not specific to ray-tracing data (e.g. converting a list of measurements into matrix form or copying the contents of a folder into another folder), it should NOT be defined within a ray-tracing module or class. 

    - If a method is common to multiple subclasses, it should be defined in the parent class. 

    - If a method can be used by any subclass of a certain parent class, it should be defined in that parent class; regardless of whether it is currently used by several of its subclasses. What matters is the scope of its *functionality*, not where it is actually invoked. 

- The repository is organized into layers, where the experiment files sit at the top layer. Upper layers can import from lower layers, but not vice versa. 

- Constants should not be hard coded deep into the codebase. 

- The naming of the methods should be appropriate and descriptive. For example:

    - Prefer imperatives for method names (e.g. `make_fixed_size_building_dataset`) rather than nouns (e.g. `fixed_size_building_dataset`).

    - Avoid redundancy in method names. For example, if a method is defined within a class named `CgMeas` and has a method `get_channel_matrix`, it is redundant to include `cg_meas` in the method name (e.g. `get_channel_matrix_from_cg_meas`).

    - The method name should not contradict the functionality or the returned values. For example, if a method is named `get_channel_matrix` but it actually returns a tuple of (channel matrix, location pairs), the name is not accurate and can be misleading.

- Methods that are not intended to be used outside of their defining class or module should be marked as private (e.g. `_read_file_with_format_x`).

- Avoid creating a new class for every new functionality. For example, instead of creating a class `WeightSlfCaviaDnn` and a class `ConvolutionalWeightSlfCaviaDnn`, it may be more appropriate to have a single class `WeightSlfCaviaDnn` with a parameter that specifies the type of architecture (e.g. "fully_connected" vs "convolutional").

- If a method is public, it should be clear from its name and docstring what its functionality is. 

- Naming conventions:
    
    - Prefixes for variables:
        ind_<something> for a counter and num_<something> for a total.
        v_ for vectors
        m_ for matrices
        t_ for tensors        
        l_ for lists
        tp_ for tuples        
        d_ for dictionaries
        b_ for booleans (or is_ or has_)
        df_ for dataframes
        If a variable can be in multiple categories (e.g. a vector and a matrix), omit the prefix. 

    - CamelCase for classes and snake_case (except for acronyms not at the beginning of the name) for functions and variables.
   
- It is OK to leave debugging dead code if it can be used later, but please place it in a function, e.g.
```python
b_check_locs_similarity = False
if b_check_locs_similarity:
    check_locs_similarity_env(l_tup_locs_cg)
```

- Public methods should have docstrings. 