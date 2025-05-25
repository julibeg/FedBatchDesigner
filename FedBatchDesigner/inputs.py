from shiny import reactive
from shiny.express import expressify, ui
from shiny.session import get_current_session
from shiny_validate import InputValidator

import params


class FloatInput:
    """
    A class to represent an input field expecting a number and returning a float.
    Wrapping an input in a class makes life easier as it keeps track of the associated
    `InputValidator` (if present) and also the session. It has methods to get and set
    the value as well as for adding validation rules.
    """

    @expressify
    def __init__(self, id, label, validator=None):
        self.id = id
        self.label = label
        self.session = get_current_session()
        self.validator = validator

        ui.input_text(self.id, label=self.label)

    def get(self):
        return getattr(self.session.input, self.id)()

    def set(self, value):
        ui.update_text(self.id, value=value, session=self.session)

    def add_rule(self, rule):
        if not self.validator:
            raise ValueError("No validator provided for this input.")
        self.validator.add_rule(self.id, rule)


class Inputs:
    """
    Shiny's `input` object is not very convenient to use in some cases (e.g. when there
    are multiple modules) and one has to keep track of the input IDs etc. This class
    makes dealing with inputs easier by keeping them all in the same place and providing
    some convenience methods.
    """

    def __init__(self):
        self._dict = {}
        self.validator = InputValidator()

        @reactive.effect
        def _():
            self.validator.enable()

    def get(self, id):
        """Get the input value for the given ID."""
        value = self._dict[id].get()
        if value is None or value == "":
            return None
        return float(self._dict[id].get())

    def set(self, id, value):
        """Set the input value for the given ID."""
        self._dict[id].set(value)

    @expressify
    def add_input(self, id, label, validator=None):
        if validator is None:
            # providing a custom validator only makes sense in a module context (i.e. if
            # none is provided, use the top-level validator)
            validator = self.validator
        self._dict[id] = FloatInput(id, label, validator=validator)

    def parse(self):
        if not self.validator.is_valid():
            # return early if self. not valid
            return
        parsed = {}
        parsed["common"] = {
            k: self.get(k) for k, v in params.common.items() if v.required
        }
        parsed["s1"] = {k: self.get(f"s1_{k}") for k in params.stage_specific.keys()}
        # use params from stage 1 in stage two unless specified otherwise
        parsed["s2"] = parsed["s1"].copy()
        for k, v in params.stage_specific.items():
            if v.stage_1_only:
                # skip stage 2 params that are only available for stage 1
                continue
            value = self.get(f"s2_{k}")
            if value is not None:
                parsed["s2"][k] = float(value)

        return parsed

    def clear(self):
        """Clear all input values."""
        for value in self._dict.values():
            value.set("")

    def add_rule(self, id, rule):
        """Add a validation rule for the input with the given ID."""
        if id not in self._dict:
            raise ValueError(f"Input with ID '{id}' does not exist.")
        self._dict[id].add_rule(rule)

    def add_validator(self, child_validator, label):
        self.validator.add_validator(child_validator, label=label)

    def all_valid(self):
        return self.validator.is_valid()
