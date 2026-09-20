

When writing code comments or documentation, please adhere to the following guidelines:

 - Prefer short direct sentences to long complicated ones

 - Avoid complicated phrasing: choosing a precise verb is good, but keep syntax simple.
 
 - Prefer (possibly nested) bullets to long paragraphs. More than 4 or 5 lines is a long paragraph. If you have a long paragraph, consider breaking it into bullets.

    - For example, when a statement follows the previous one, use `->` as the bullet prefix. For example:

        # Special case: the patch covers the whole extent of the
        # measurements along this axis.
        # -> Then `patch_side_len` is (approximately) `coord_max -
        #   coord_min`, so `lim_blc_max` is (approximately)
        #   `coord_min`.
        # -> There is then only one admissible corner: `coord_min`.


    - Use bullets for cases. This applies, in particular, to **docstrings**. For example, the following explanation 

    ```
        - `fallback`: what `self.estimate` does when the estimation raises
          `InsufficientObservationsForEstimationError`. If None, the
          exception propagates. If a `MapEstimator`, that estimator is
          invoked instead on the same observations and test locations. If a
          vector of length num_channels, the fallback estimator is
          `ConstantMapEstimator(fallback)`, i.e. the estimate takes the
          given value at every location of the corresponding channel.
    ```
    
    should be written as:

    ```
        - `fallback`: specifies what `self.estimate` does when the estimation raises
          `InsufficientObservationsForEstimationError`. 
          
            - If None, the exception propagates. 
            
            - If a `MapEstimator`, that estimator is invoked instead.
            
            - If a vector of length num_channels, the fallback estimator 
            `ConstantMapEstimator(fallback)` is used. 
    ```