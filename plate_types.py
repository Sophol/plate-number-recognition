"""Plate type vocabulary shared by the inference and event workers.

These strings are written to plate_reads.plate_type and read back when deciding
a gate action, so both workers must agree on them. They live in a module with no
dependencies because the alternatives are worse: putting them in db.models drags
SQLAlchemy into the pure inference pipeline, and putting them in the inference
worker makes the event worker depend on it.
"""

PRIVATE_CAR = "private_car"
MOTORCYCLE = "motorcycle"

# A plate whose number is replaced by a Khmer name -- rare, and in practice
# reserved for VIP vehicles. No OCR reads it and no format regex matches it, so
# it is a distinct plate type rather than a failed read.
VANITY = "vanity"
