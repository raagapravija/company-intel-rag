"""Neo4j driver helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from neo4j import Driver, GraphDatabase, RoutingControl

from company_intel.config import Settings, get_settings


def make_driver(settings: Settings | None = None, *, readonly: bool = False) -> Driver:
    """Create a driver. `readonly=True` uses the restricted user (for Text2Cypher)."""
    s = settings or get_settings()
    if readonly:
        auth = (s.neo4j_readonly_user, s.neo4j_readonly_password.get_secret_value())
    else:
        auth = (s.neo4j_user, s.neo4j_password.get_secret_value())
    return GraphDatabase.driver(s.neo4j_uri, auth=auth)


@contextmanager
def driver_session(settings: Settings | None = None, *, readonly: bool = False) -> Iterator[Driver]:
    driver = make_driver(settings, readonly=readonly)
    try:
        driver.verify_connectivity()
        yield driver
    finally:
        driver.close()


def run_read(driver: Driver, query: str, **params: Any) -> list[dict[str, Any]]:
    s = get_settings()
    records, _, _ = driver.execute_query(
        query, params, database_=s.neo4j_database, routing_=RoutingControl.READ
    )
    return [r.data() for r in records]


def run_write(driver: Driver, query: str, **params: Any) -> None:
    s = get_settings()
    driver.execute_query(query, params, database_=s.neo4j_database)
