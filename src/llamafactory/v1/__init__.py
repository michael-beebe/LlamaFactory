# Copyright 2025 the LlamaFactory authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Polyfill enum.StrEnum for Python 3.10 (it landed in 3.11). LlamaFactory v1
# uses `from enum import StrEnum` in several modules and is officially
# Python 3.11+, but we want to run it inside our 3.10 conda env that has
# torch + torchcomms + mscclpp built. This must execute before any submodule
# is imported.
import enum as _enum
import sys as _sys
import typing as _typing

if not hasattr(_enum, "StrEnum"):
    class StrEnum(str, _enum.Enum):  # type: ignore[no-redef]
        def __new__(cls, value, *args, **kwargs):
            if not isinstance(value, str):
                raise TypeError(f"StrEnum values must be str, got {type(value)}")
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj

        def __str__(self):
            return self.value

    _enum.StrEnum = StrEnum  # type: ignore[attr-defined]

# Polyfill typing.NotRequired / typing.Required / typing.Self (3.11+) from
# typing_extensions which is always available in the conda env.
import typing_extensions as _te
for _attr in ("NotRequired", "Required", "Self", "Never", "LiteralString",
              "Unpack", "TypeVarTuple", "ParamSpec"):
    if not hasattr(_typing, _attr) and hasattr(_te, _attr):
        setattr(_typing, _attr, getattr(_te, _attr))
