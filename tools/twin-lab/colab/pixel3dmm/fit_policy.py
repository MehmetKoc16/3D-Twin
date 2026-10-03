"""Bounded mini-batches and comparable per-view joint optimization budgets."""

import math


def fit_budget(view_count: int, iters: int, global_iters: int, max_batch_size: int) -> dict:
    if not isinstance(view_count, int) or not 1 <= view_count <= 12:
        raise ValueError("Fit requires 1 through 12 usable views")
    if any(not isinstance(value, int) or value < 1 for value in (iters, global_iters, max_batch_size)):
        raise ValueError("Iteration and batch limits must be positive integers")
    batch = min(view_count, max_batch_size)
    # GLOBAL_ITERS remains a floor; its reference is the tested two-view run.
    joint = max(global_iters, math.ceil(global_iters * view_count / (2 * batch)))
    effective_joint = joint if view_count > 1 else 0
    return {"usableViews": view_count, "batchSize": batch, "onlineItersPerView": iters,
            "jointIters": joint, "jointItersFloor": global_iters, "referenceViews": 2,
            "expectedJointUpdatesPerView": effective_joint * batch / view_count,
            "onlinePlusJointIterations": view_count * iters + effective_joint,
            "sampleWorkUnits": view_count * iters + effective_joint * batch}
