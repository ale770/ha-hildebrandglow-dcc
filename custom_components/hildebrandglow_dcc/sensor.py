"""Platform for sensor integration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, date
import logging
from operator import itemgetter
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

import requests

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.models import StatisticData, StatisticMetaData
from homeassistant.components.recorder.statistics import (
    async_add_external_statistics,
    get_last_statistics,
)
from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.event import async_track_time_change
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)
from homeassistant.util import dt as dt_util

from .const import DOMAIN

gql_transport = None
gql_client = None

token_query = """mutation {{
	obtainKrakenToken(input: {{ APIKey: "{api_key}" }}) {{
	    token
	}}
}}"""

account_query = """query{{
    account(
        accountNumber: "{acc_number}"
    ) {{
    electricityAgreements(active: true) {{
        validFrom
        validTo
        meterPoint {{
            meters(includeInactive: false) {{
                smartDevices {{
                    deviceId
                }}
            }}
            mpan
        }}
        tariff {{
            ... on HalfHourlyTariff {{
                id
                productCode
                tariffCode
                productCode
                standingCharge
                }}
            }}
        }}
    }}
}}"""

_LOGGER = logging.getLogger(__name__)
UPDATE_MINUTES = [1, 31]
BASE_URL = "https://api.octopus.energy/v1"


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: Callable
) -> bool:
    """Set up the sensor platform."""
    entities: list[SensorEntity] = []
    meters: dict[str, SensorEntity] = {}

    # Get API object from the config flow.
    glowmarkt = hass.data[DOMAIN][entry.entry_id]
    api_key = entry.data["api_key"]
    account_number = entry.data["account_number"]

    # Gather all virtual entities on the account
    virtual_entities: dict = {}
    try:
        virtual_entities = await hass.async_add_executor_job(
            glowmarkt.get_virtual_entities
        )
        _LOGGER.debug("Successful GET to %svirtualentity", glowmarkt.url)
    except requests.Timeout as ex:
        _LOGGER.error("Timeout: %s", ex)
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error("Cannot connect: %s", ex)
    except Exception as ex:  # pylint: disable=broad-except
        if "Request failed" in str(ex):
            _LOGGER.error(
                "Non-200 Status Code. The Glow API may be experiencing issues"
            )
        else:
            _LOGGER.exception("Unexpected exception: %s. Please open an issue", ex)

    for virtual_entity in virtual_entities:
        # Gather all resources for each virtual entity
        resources: dict = {}
        try:
            resources = await hass.async_add_executor_job(virtual_entity.get_resources)
            _LOGGER.debug(
                "Successful GET to %svirtualentity/%s/resources",
                glowmarkt.url,
                virtual_entity.id,
            )
        except requests.Timeout as ex:
            _LOGGER.error("Timeout: %s", ex)
        except requests.exceptions.ConnectionError as ex:
            _LOGGER.error("Cannot connect: %s", ex)
        except Exception as ex:  # pylint: disable=broad-except
            if "Request failed" in str(ex):
                _LOGGER.error(
                    "Non-200 Status Code. The Glow API may be experiencing issues"
                )
            else:
                _LOGGER.exception("Unexpected exception: %s. Please open an issue", ex)

        # Loop through all resources and create sensors
        for resource in resources:
            if resource.classifier in ["electricity.consumption", "gas.consumption"]:
                usage_sensor = HistoricalUsageSensor(
                    hass, virtual_entity, resource, api_key, account_number
                )
                entities.append(usage_sensor)
                meters[resource.classifier] = usage_sensor

                # Schedule the sensor to update every day at 12:00 PM.
                async_track_time_change(
                    hass,
                    usage_sensor.async_update_callback,
                    minute=UPDATE_MINUTES,
                    second=0,
                )

                coordinator = TariffCoordinator(hass, resource)
                standing_sensor = Standing(coordinator, resource, virtual_entity)
                entities.append(standing_sensor)
                rate_sensor = Rate(coordinator, resource, virtual_entity)
                entities.append(rate_sensor)

                async_track_time_change(
                    hass,
                    coordinator.async_update_callback,
                    minute=UPDATE_MINUTES,
                    second=0,
                )

    async_add_entities(entities, update_before_add=True)
    return True


def supply_type(resource) -> str:
    """Return the supply type based on resource classifier."""
    if "electricity.consumption" in resource.classifier:
        return "electricity"
    if "gas.consumption" in resource.classifier:
        return "gas"
    _LOGGER.error("Unknown classifier: %s. Please open an issue", resource.classifier)
    return "unknown"


def device_name(resource, virtual_entity) -> str:
    """Return a device name based on the resource and virtual entity."""
    supply = supply_type(resource)
    if virtual_entity.name:
        return f"{virtual_entity.name} Smart {supply.capitalize()} Meter"
    return f"Smart {supply.capitalize()} Meter"


async def daily_data(
    hass: HomeAssistant, resource, t_from: datetime = None
) -> list[tuple[datetime, float]] | None:
    """Return hourly readings for a daily period from the API."""

    t_to = await hass.async_add_executor_job(
        resource.round,
        (datetime.now() - timedelta(hours=1)).replace(minute=59, second=59),
        "PT1M",
    )
    try:
        await hass.async_add_executor_job(resource.catchup)
        _LOGGER.debug("Successful GET to /resource/%s/catchup", resource.id)
    except requests.Timeout as ex:
        _LOGGER.error("Timeout during catchup for resource %s: %s", resource.id, ex)
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error(
            "Cannot connect during catchup for resource %s: %s", resource.id, ex
        )
    except Exception as ex:
        if "Request failed" in str(ex):
            _LOGGER.warning(
                "Non-200 Status Code during catchup for resource %s; the API may be having issues",
                resource.id,
            )
        else:
            _LOGGER.exception(
                "Unexpected exception during catchup for resource %s: %s. Please open an issue",
                resource.id,
                ex,
            )

    try:
        readings = await hass.async_add_executor_job(
            resource.get_readings, t_from, t_to, "PT1H", "sum", True
        )
        _LOGGER.debug("Successfully got hourly readings for resource %s", resource.id)
        return readings
    except requests.Timeout as ex:
        _LOGGER.error("Timeout fetching readings for resource %s: %s", resource.id, ex)
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error(
            "Cannot connect fetching readings for resource %s: %s", resource.id, ex
        )
    except Exception as ex:
        if "Request failed" in str(ex):
            _LOGGER.warning(
                "Non-200 Status Code when fetching readings for resource %s",
                resource.id,
            )
        else:
            _LOGGER.exception(
                "Unexpected exception fetching readings for resource %s: %s. Please open an issue",
                resource.id,
                ex,
            )
    return None


def _generate_statistics_from_readings(
    readings: list[tuple[datetime, float]],
    cumulative_start: float = 0.0,
    unit_rates: list[dict] = None,
) -> list[StatisticData]:
    """Convert a list of (datetime, reading) entries into StatisticData entries."""
    sorted_readings = sorted(readings, key=lambda x: x[0])
    cumulative = cumulative_start
    stats: list[StatisticData] = []
    for dt_obj, elem in sorted_readings:
        # Normalize the start timestamp to the hour
        hour_ts = dt_obj.replace(minute=0, second=0, microsecond=0)
        # convert to timestamp
        read_time = hour_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        value = elem.value
        if unit_rates is not None:
            # Find the unit rate that applies to this hour
            matching_rate = next(
                rate
                for rate in unit_rates
                if rate["valid_from"] < read_time <= rate["valid_to"]
            )
            cost = float("{:.4f}".format(value * matching_rate["value_inc_vat"]))
            value = cost / 100
        cumulative += value
        stats.append(
            StatisticData(
                start=dt_util.as_utc(hour_ts),
                state=value,
                sum=cumulative,
            )
        )
    return stats


def _discard_after_last_non_zero_reading(readings):
    # Iterate through the readings in reverse order
    for i in range(len(readings) - 1, -1, -1):
        if readings[i][1].value != 0:
            # Slice the list to exclude the last non-zero reading
            return readings[:i]
    # If all readings are zero, return an empty list
    return []


async def tariff_data(hass: HomeAssistant, resource) -> dict | None:
    """Get tariff data from the API."""
    try:
        tariff = await hass.async_add_executor_job(resource.get_tariff)
        _LOGGER.debug("Successful GET for tariff data of resource %s", resource.id)
        return tariff
    except requests.Timeout as ex:
        _LOGGER.error(
            "Timeout retrieving tariff data for resource %s: %s", resource.id, ex
        )
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error(
            "Cannot connect retrieving tariff data for resource %s: %s", resource.id, ex
        )
    except Exception as ex:
        if "Request failed" in str(ex):
            _LOGGER.warning(
                "Non-200 Status Code when fetching tariff data for resource %s",
                resource.id,
            )
        else:
            _LOGGER.exception(
                "Unexpected exception retrieving tariff data for resource %s: %s",
                resource.id,
                ex,
            )
    return None


def setup_gql(token):
    global gql_transport, gql_client
    gql_transport = AIOHTTPTransport(
        url=f"{BASE_URL}/graphql/", headers={"Authorization": f"{token}"}
    )
    gql_client = Client(transport=gql_transport, fetch_schema_from_transport=True)


def get_token(api_key):
    transport = AIOHTTPTransport(url=f"{BASE_URL}/graphql/")
    client = Client(transport=transport, fetch_schema_from_transport=True)
    query = gql(token_query.format(api_key=api_key))
    result = client.execute(query)
    return result["obtainKrakenToken"]["token"]


def get_acc_info(account_number):
    query = gql(account_query.format(acc_number=account_number))
    result = gql_client.execute(query)
    tariff_code = next(
        agreement["tariff"]["tariffCode"]
        for agreement in result["account"]["electricityAgreements"]
        if "tariffCode" in agreement["tariff"]
    )
    region_code = tariff_code[-1]

    if "GO" in tariff_code:
        current_tariff = "GO"
    elif "AGILE" in tariff_code:
        current_tariff = "AGILE"
    else:
        raise Exception(f"ERROR: Unknown tariff code: {tariff_code}")

    return current_tariff, region_code


def rest_query(url):
    response = requests.get(url)
    if response.ok:
        data = response.json()
        return data
    else:
        raise Exception(
            f"ERROR: rest_query failed querying `{url}` with {response.status_code}"
        )


def get_potential_tariff_rates(tariff, region_code, lookup_date_from):
    all_products = rest_query(f"{BASE_URL}/products")
    tariff_code = next(
        product["code"]
        for product in all_products["results"]
        if product["display_name"]
        == ("Agile Octopus" if tariff == "AGILE" else "Octopus Go")
        and product["direction"] == "IMPORT"
        and product["brand"] == "OCTOPUS_ENERGY"
    )
    # Residential tariffs are always E-1R (i think, lol)
    product_code = f"E-1R-{tariff_code}-{region_code}"
    # substract 1 day from lookup_date_from
    lookup_date_from = lookup_date_from - timedelta(days=1)

    unit_rates = []
    while lookup_date_from.date() <= date.today():
        unit_rates += rest_query(
            f"{BASE_URL}/products/{tariff_code}/electricity-tariffs/{product_code}/standard-unit-rates/?period_from={lookup_date_from.year}-{lookup_date_from.month}-{lookup_date_from.day}T00:00:00Z&period_to={lookup_date_from.year}-{lookup_date_from.month}-{lookup_date_from.day}T23:59:59Z"
        )["results"]
        lookup_date_from += timedelta(days=1)

    return unit_rates


class HistoricalUsageSensor(SensorEntity):
    """Sensor for hourly consumption (usage) data."""

    _attr_state_class = SensorStateClass.TOTAL
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR

    def __init__(
        self, hass: HomeAssistant, virtual_entity, resource, api_key, account_number
    ) -> None:
        """Initialize the consumption sensor."""
        self.hass = hass
        self._virtual_entity = virtual_entity
        self._resource = resource
        self._api_key = api_key
        self._account_number = account_number
        supply = supply_type(resource)
        self._attr_unique_id = f"{supply}_consumption_{resource.id}"
        self._attr_name = (
            f"{virtual_entity.name} {supply.capitalize()} Consumption"
            if virtual_entity.name
            else f"{supply.capitalize()} Consumption"
        )
        self._state: float | None = None
        self._attr_should_poll = False

    @property
    def state(self) -> float | None:
        """Return the most recent consumption value."""
        return self._state

    @property
    def icon(self) -> str | None:
        """Icon to use in the frontend."""
        # Only the gas usage sensor needs an icon as the others inherit from their device class.
        if self._resource.classifier == "gas.consumption":
            return "mdi:fire"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information for the consumption sensor."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._resource.id)},
            manufacturer="Hildebrand",
            model="Glow (DCC)",
            name=device_name(self._resource, self._virtual_entity),
        )

    @callback
    async def async_update_callback(self, ts) -> None:
        """Callback triggered by time change to update the sensor and inject statistics."""
        await self.async_update()
        self.async_write_ha_state()

    async def async_update(self) -> None:
        """Fetch hourly consumption data and update sensor state and statistics."""
        t_from = None
        stat_id = f"{DOMAIN}:{supply_type(self._resource)}_consumption"
        cost_stat_id = f"{DOMAIN}:{supply_type(self._resource)}_cost"

        try:
            # Look up the most recent statistics data. This lookup runs in the executor.
            last_stats = await get_instance(self.hass).async_add_executor_job(
                get_last_statistics, self.hass, 1, stat_id, True, {"sum"}
            )
            # If a previous value exists, use its "sum" as the starting cumulative.
            if len(last_stats.get(stat_id, [])) > 0:
                last_stats = last_stats[stat_id]
                last_stats = sorted(last_stats, key=itemgetter("start"), reverse=False)[
                    0
                ]

            # Look up the most recent statistics data. This lookup runs in the executor.
            cost_last_stats = await get_instance(self.hass).async_add_executor_job(
                get_last_statistics, self.hass, 1, cost_stat_id, True, {"sum"}
            )
            # If a previous value exists, use its "sum" as the starting cumulative.
            if len(cost_last_stats.get(cost_stat_id, [])) > 0:
                cost_last_stats = cost_last_stats[cost_stat_id]
                cost_last_stats = sorted(
                    cost_last_stats, key=itemgetter("start"), reverse=False
                )[0]
        except AttributeError:
            last_stats = None
            cost_last_stats = None
        if not last_stats:
            # First time lets insert last 30 days of data
            t_from = await self.hass.async_add_executor_job(
                self._resource.round, datetime.now() - timedelta(days=30), "P1D"
            )

        if t_from is None:
            t_from = await self.hass.async_add_executor_job(
                self._resource.round, datetime.now() - timedelta(hours=24), "P1D"
            )

        readings = await daily_data(self.hass, self._resource, t_from)
        readings = _discard_after_last_non_zero_reading(readings)

        if last_stats is not None and last_stats.get("sum") is not None:
            initial_cumulative = last_stats["sum"]
            # Discard all readings before last_stats["start"].
            start_ts = dt_util.as_utc(datetime.fromtimestamp(last_stats.get("start")))
            readings = [r for r in readings if r[0] > start_ts]
        else:
            initial_cumulative = 0.0

        if cost_last_stats is not None and cost_last_stats.get("sum") is not None:
            cost_initial_cumulative = cost_last_stats["sum"]
            # Discard all readings before cost_last_stats["start"].
            start_ts = dt_util.as_utc(
                datetime.fromtimestamp(cost_last_stats.get("start"))
            )
        else:
            cost_initial_cumulative = 0.0

        if len(readings) == 0:
            return

        token = await get_instance(self.hass).async_add_executor_job(
            get_token, self._api_key
        )
        await get_instance(self.hass).async_add_executor_job(setup_gql, token)
        (curr_tariff, region_code) = await get_instance(
            self.hass
        ).async_add_executor_job(get_acc_info, self._account_number)
        (unit_rates) = await get_instance(self.hass).async_add_executor_job(
            get_potential_tariff_rates, curr_tariff, region_code, t_from
        )

        # Generate new StatisticData entries using the previous cumulative sum.
        stats = _generate_statistics_from_readings(
            readings, cumulative_start=initial_cumulative
        )

        cost_stats = _generate_statistics_from_readings(
            readings, cumulative_start=cost_initial_cumulative, unit_rates=unit_rates
        )
        self._state = round(readings[-1][1].value, 2)

        metadata = StatisticMetaData(
            has_mean=False,
            has_sum=True,
            name=(
                f"{self._virtual_entity.name} {supply_type(self._resource).capitalize()} Consumption"
                if self._virtual_entity.name
                else f"{supply_type(self._resource).capitalize()} Consumption"
            ),
            source=DOMAIN,
            statistic_id=stat_id,
            unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        )

        cost_metadata = StatisticMetaData(
            has_mean=False,
            has_sum=True,
            name=(
                f"{self._virtual_entity.name} {supply_type(self._resource).capitalize()} Cost"
                if self._virtual_entity.name
                else f"{supply_type(self._resource).capitalize()} Cost"
            ),
            source=DOMAIN,
            statistic_id=cost_stat_id,
            unit_of_measurement="GBP",
        )
        # Register the new statistics with Home Assistant.
        async_add_external_statistics(self.hass, metadata, stats)
        async_add_external_statistics(self.hass, cost_metadata, cost_stats)


class TariffCoordinator(DataUpdateCoordinator):
    """Data update coordinator for tariff sensors."""

    def __init__(self, hass: HomeAssistant, resource) -> None:
        """Initialize the tariff coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            # Name of the data. For logging purposes.
            name="tariff",
            # Polling interval. Will only be polled if there are subscribers.
            update_interval=None,
        )

        self._resource = resource

    @callback
    async def async_update_callback(self, ts) -> None:
        """Callback triggered by time change to update the sensor and inject statistics."""
        await self.async_request_refresh()

    async def _async_update_data(self):
        """Fetch data from tariff API endpoint."""
        return await tariff_data(self.hass, self._resource)


class Standing(CoordinatorEntity, SensorEntity):
    """An entity using CoordinatorEntity.

    The CoordinatorEntity class provides:
      should_poll
      async_update
      async_added_to_hass
      available

    """

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_has_entity_name = True
    _attr_name = "Standing charge"
    _attr_native_unit_of_measurement = "GBP"
    _attr_entity_registry_enabled_default = (
        False  # Don't enable by default as less commonly used
    )

    def __init__(self, coordinator, resource, virtual_entity) -> None:
        """Pass coordinator to CoordinatorEntity."""
        super().__init__(coordinator)

        self._attr_unique_id = resource.id + "-tariff"

        self._resource = resource
        self._virtual_entity = virtual_entity

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if self.coordinator.data:
            value = (
                float(self.coordinator.data.current_rates.standing_charge.value) / 100
            )
            self._attr_native_value = round(value, 4)
            self.async_write_ha_state()

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._resource.id)},
            manufacturer="Hildebrand",
            model="Glow (DCC)",
            name=device_name(self._resource, self._virtual_entity),
        )


class Rate(CoordinatorEntity, SensorEntity):
    """An entity using CoordinatorEntity.

    The CoordinatorEntity class provides:
      should_poll
      async_update
      async_added_to_hass
      available

    """

    _attr_device_class = None
    _attr_has_entity_name = True
    _attr_icon = (
        "mdi:cash-multiple"  # Need to provide an icon as doesn't have a device class
    )
    _attr_name = "Rate"
    _attr_native_unit_of_measurement = "GBP/kWh"
    _attr_entity_registry_enabled_default = (
        False  # Don't enable by default as less commonly used
    )

    def __init__(self, coordinator, resource, virtual_entity) -> None:
        """Pass coordinator to CoordinatorEntity."""
        super().__init__(coordinator)

        self._attr_unique_id = resource.id + "-rate"

        self._resource = resource
        self._virtual_entity = virtual_entity

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        if self.coordinator.data:
            value = float(self.coordinator.data.current_rates.rate.value) / 100
            self._attr_native_value = round(value, 4)
            self.async_write_ha_state()

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._resource.id)},
            manufacturer="Hildebrand",
            model="Glow (DCC)",
            name=device_name(self._resource, self._virtual_entity),
        )
