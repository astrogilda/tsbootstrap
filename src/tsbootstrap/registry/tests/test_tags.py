"""Tests for tag register an tag functionality."""

from tsbootstrap.registry._tags import OBJECT_TAG_REGISTER


def test_tag_register_type():
    """Test the specification of the tag register. See _tags for specs."""
    if not isinstance(OBJECT_TAG_REGISTER, list):
        raise TypeError("OBJECT_TAG_REGISTER is not a list.")
    if not all(isinstance(tag, tuple) for tag in OBJECT_TAG_REGISTER):
        raise TypeError("Not all elements in OBJECT_TAG_REGISTER are tuples.")

    for tag in OBJECT_TAG_REGISTER:
        if len(tag) != 4:
            raise ValueError("Tag does not have 4 elements.")
        if not isinstance(tag[0], str):
            raise TypeError("Tag name is not a string.")
        if not isinstance(tag[1], (str, list)):
            raise TypeError("Tag type is not a string or list.")
        if isinstance(tag[1], list) and not all(
            isinstance(x, str) for x in tag[1]
        ):
            raise TypeError("Not all elements in tag type list are strings.")
        if not isinstance(tag[2], (str, tuple)):
            raise TypeError("Tag description is not a string or tuple.")
        if isinstance(tag[2], tuple):
            if not len(tag[2]) == 2:
                raise ValueError(
                    "Tag description tuple does not have 2 elements."
                )
            if not isinstance(tag[2][0], str):
                raise TypeError(
                    "Tag description tuple first element is not a string."
                )
            if not isinstance(tag[2][1], (list, str)):
                raise TypeError(
                    "Tag description tuple second element is not a list or string."
                )
            if isinstance(tag[2][1], list) and not all(
                isinstance(x, str) for x in tag[2][1]
            ):
                raise TypeError(
                    "Not all elements in tag description list are strings."
                )
        if not isinstance(tag[3], str):
            raise TypeError("Tag source is not a string.")
