from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4


class ModelInfo(BaseModel):
    version: str
    timestamp: str


class ModelInput(BaseModel):
    uuid: str = Field(default=str(uuid4()))
    account_amount_added_12_24m: float = Field(..., ge=0.0)
    account_days_in_dc_12_24m: float = Field(..., ge=0.0)
    account_days_in_rem_12_24m: float = Field(..., ge=0 - 0)
    account_days_in_term_12_24m: float = Field(..., ge=0 - 0)
    account_incoming_debt_vs_paid_0_24m: float = Field(..., ge=0 - 0)
    account_status: int = Field(..., ge=0)
    account_worst_status_0_3m: int = Field(..., ge=0)
    account_worst_status_12_24m: int = Field(..., ge=0)
    account_worst_status_3_6m: int = Field(..., ge=0)
    account_worst_status_6_12m: int = Field(..., ge=0)
    age: int = Field(..., ge=0, le=100)
    avg_payment_span_0_12m: float = Field(..., ge=0.0)
    avg_payment_span_0_3m: float = Field(..., ge=0.0)
    merchant_category: str = Field(default="Missing")
    merchant_group: str = Field(default="Missing")
    has_paid: bool
    max_paid_inv_0_12m: float = Field(..., ge=0.0)
    max_paid_inv_0_24m: float = Field(..., ge=0.0)
    name_in_email: str = Field(default="Missing")
    num_active_div_by_paid_inv_0_12m: int = Field(..., ge=0)
    num_active_inv: int = Field(..., ge=0)
    num_arch_dc_0_12m: int = Field(..., ge=0)
    num_arch_dc_12_24m: int = Field(..., ge=0)
    num_arch_ok_0_12m: int = Field(..., ge=0)
    num_arch_ok_12_24m: int = Field(..., ge=0)
    num_arch_rem_0_12m: int = Field(..., ge=0)
    num_arch_written_off_0_12m: int = Field(..., ge=0)
    num_arch_written_off_12_24m: int = Field(..., ge=0)
    num_unpaid_bills: int = Field(..., ge=0)
    status_last_archived_0_24m: int = Field(..., ge=0)
    status_2nd_last_archived_0_24m: int = Field(..., ge=0)
    status_3rd_last_archived_0_24m: int = Field(..., ge=0)
    status_max_archived_0_6_months: int = Field(..., ge=0)
    status_max_archived_0_12_months: int = Field(..., ge=0)
    status_max_archived_0_24_months: int = Field(..., ge=0)
    recovery_debt: float = Field(..., ge=0.0)
    sum_capital_paid_account_0_12m: float = Field(..., ge=0.0)
    sum_capital_paid_account_12_24m: float = Field(..., ge=0.0)
    sum_paid_inv_0_12m: float = Field(..., ge=0.0)
    time_hours: float = Field(..., ge=0.0)
    worst_status_active_inv: int = Field(..., ge=0.0)


class ModelOutPut(BaseModel):
    uuid: str
    probability_default: float = Field(q3=0.0, le=1.0)
