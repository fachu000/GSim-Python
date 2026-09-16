

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