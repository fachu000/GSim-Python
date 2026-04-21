from dataclasses import dataclass, field
from typing import Generic

from .defs import RawInputType, InputType, OutputType, TargetType


@dataclass
class AdaptationSpec:
    """Flags that describe in which phase of the pipeline a DataAdapter method
    is being called.

    Args:
        preprocess_only: True iff called from
            ``load_or_create_preprocessed_dataset``.
        input_already_preprocessed: True iff the input has already been
            processed by a previous call with ``preprocess_only=True`` (i.e. the
            dataset was loaded from disk after preprocessing).
        inference: True iff called from ``NeuralNet.predict``; False when
            called from ``fit``, ``evaluate``, or
            ``load_or_create_preprocessed_dataset``.
    """
    preprocess_only: bool = False
    input_already_preprocessed: bool = False
    inference: bool = False


class DataAdapter(Generic[RawInputType, InputType]):
    """Base class for data adapters.

    A DataAdapter transforms raw data at several points in the NeuralNet
    pipeline.  All methods receive an :class:`AdaptationSpec` that describes
    the current pipeline phase so that subclasses can conditionally skip or
    apply work (e.g. avoid recomputing cached features).

    The default implementation of every method is the identity.
    
    See :ref:`usage.md` for examples of how to use this class.
    """

    def adapt_input(self, raw_input: RawInputType,
                    spec: AdaptationSpec) -> InputType:
        """Transform a single raw input before normalization and forward pass.
        This function can be used e.g. for feature extraction, data
        augmentation, or self-supervised (input,target) pair formation.

        Args:
            raw_input: A single raw input. spec: Flags describing the current
            pipeline phase.

        Returns:
            Processed input.
        """
        return raw_input

    def adapt_target(self, target: TargetType,
                     spec: AdaptationSpec) -> TargetType:
        """Transform a single target.

        Args:
            target: A single target.
            spec: Flags describing the current pipeline phase.

        Returns:
            Processed target.
        """
        return target

    def adapt_dataset_item(self, item, spec: AdaptationSpec):
        """Transform a ``(raw_input, target)`` pair.

        The default implementation splits ``item`` into input and target,
        applies :meth:`adapt_input` and :meth:`adapt_target` independently,
        and returns the resulting pair.  Override when input and target must
        be transformed jointly (e.g. random crop applied to both image and
        mask).

        Args:
            item: A ``(raw_input, target)`` pair.
            spec: Flags describing the current pipeline phase.

        Returns:
            A ``(adapted_input, adapted_target)`` pair.
        """
        raw_input, target = item
        return self.adapt_input(raw_input,
                                spec), self.adapt_target(target, spec)

    def adapt_output(self, output: OutputType,
                     spec: AdaptationSpec) -> OutputType:
        """Transform a single network output (called during inference).

        Args:
            output: A single network output (after optional unnormalization).
            spec: Flags describing the current pipeline phase.

        Returns:
            Post-processed output.
        """
        return output

    def get_no_targets(self, inner_dataset_has_no_targets: bool,
                       spec: AdaptationSpec) -> bool:
        """Declare whether the dataset that results from adaptation has no
        targets.

        Override when the adapter synthesizes targets from inputs (e.g.
        self-supervised pair formation inside ``adapt_input``), so that the rest
        of the pipeline (collate, loss) sees the correct batch structure.

        Args:
            inner_dataset_has_no_targets: Whether the dataset being wrapped
                contains only inputs (no targets).
            spec: Flags describing the current pipeline phase.

        Returns:
            Whether the dataset resulting from adaptation has no targets.
        """
        return inner_dataset_has_no_targets
