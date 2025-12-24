import re


ORDER_ID_RE = re.compile(
    r"(order|tracking)\s*(id|number)?\s*[:#]?\s*(\d{9})(?:\.0)?",
    re.I
)

BASIC_FIELDS = {
    "order_number",
    "order_date",
    "carrier",
    "shipping_method",
    "tracking_number",
}

DETAILED_FIELDS = {
    "to_name",
    "zip",
    "state",
    "country",
    "warehouse",
    "tpl_customer",
    "size_dimensions",
    "weight_oz",
    "final_amount",
}

ZIP_RE = re.compile(r"\b\d{5}\b")
WEIGHT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg|kgs|kilograms|lb|lbs|pounds|oz|ounces)",
    re.I
)
