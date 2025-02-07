"""Platform for sensor integration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
import itertools
import logging
import statistics

from homeassistant_historical_sensor import (
    HistoricalSensor,
    HistoricalState,
    PollUpdateMixin,
)
import requests

from homeassistant.components.recorder.models import StatisticData, StatisticMetaData
from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import UnitOfEnergy
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)
from homeassistant.util import dt as dt_util

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)
SCAN_INTERVAL = timedelta(minutes=5)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: Callable
) -> bool:
    """Set up the sensor platform."""
    entities: list = []
    meters: dict = {}

    # Get API object from the config flow
    glowmarkt = hass.data[DOMAIN][entry.entry_id]

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
                historical_usage_sensor = HistoricalUsage(
                    hass, resource, virtual_entity
                )
                entities.append(historical_usage_sensor)
                meters[resource.classifier] = historical_usage_sensor

                coordinator = TariffCoordinator(hass, resource)
                standing_sensor = Standing(coordinator, resource, virtual_entity)
                entities.append(standing_sensor)
                rate_sensor = Rate(coordinator, resource, virtual_entity)
                entities.append(rate_sensor)

        # Cost sensors must be created after usage sensors as they reference them as a meter
        for resource in resources:
            if resource.classifier == "gas.consumption.cost":
                historical_cost_sensor = HistoricalCost(hass, resource, virtual_entity)
                historical_cost_sensor.meter = meters["gas.consumption"]
                entities.append(historical_cost_sensor)
            elif resource.classifier == "electricity.consumption.cost":
                historical_cost_sensor = HistoricalCost(hass, resource, virtual_entity)
                historical_cost_sensor.meter = meters["electricity.consumption"]
                entities.append(historical_cost_sensor)

    async_add_entities(entities, update_before_add=True)

    return True


def supply_type(resource) -> str:
    """Return supply type."""
    if "electricity.consumption" in resource.classifier:
        return "electricity"
    if "gas.consumption" in resource.classifier:
        return "gas"
    _LOGGER.error("Unknown classifier: %s. Please open an issue", resource.classifier)
    return "unknown"


def device_name(resource, virtual_entity) -> str:
    """Return device name. Includes name of virtual entity if it exists."""
    supply = supply_type(resource)
    if virtual_entity.name is not None:
        name = f"{virtual_entity.name} smart {supply} meter"
    else:
        name = f"Smart {supply} meter"
    return name


async def should_update() -> bool:
    """Check if time is between 0-5 or 30-35 minutes past the hour."""
    minutes = datetime.now().minute
    if (1 <= minutes <= 5) or (31 <= minutes <= 35):
        return True
    return False


def discard_after_last_non_zero_reading(readings):
    # Iterate through the readings in reverse order
    for i in range(len(readings) - 1, -1, -1):
        if readings[i][1].value != 0:
            # Slice the list to exclude the last non-zero reading
            return readings[:i]
    # If all readings are zero, return an empty list
    return []


async def daily_data(
    hass: HomeAssistant, resource, t_from: datetime = None
) -> (float, str):
    """Get daily usage from the API."""
    if t_from is None:
        t_from = await hass.async_add_executor_job(
            resource.round, datetime.now() - timedelta(hours=12), "P1D"
        )

    t_to = await hass.async_add_executor_job(
        resource.round,
        (datetime.now() - timedelta(hours=1)).replace(minute=59, second=59),
        "PT1M",
    )
    try:
        await hass.async_add_executor_job(resource.catchup)
        _LOGGER.debug(
            "Successful GET to https://api.glowmarkt.com/api/v0-1/resource/%s/catchup",
            resource.id,
        )
    except requests.Timeout as ex:
        _LOGGER.error("Timeout: %s", ex)
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error("Cannot connect: %s", ex)
    except Exception as ex:  # pylint: disable=broad-except
        if "Request failed" in str(ex):
            _LOGGER.warning(
                "Non-200 Status Code. The Glow API may be experiencing issues"
            )
        else:
            _LOGGER.exception("Unexpected exception: %s. Please open an issue", ex)

    try:
        # difference in days between t_from and t_to
        readings = await hass.async_add_executor_job(
            resource.get_readings, t_from, t_to, "PT1H", "sum", True
        )
        _LOGGER.debug("Successfully got daily usage for resource id %s", resource.id)
        # Last reading may not be complete, so discard.
        filtered_readings = discard_after_last_non_zero_reading(readings)
        _LOGGER.debug("Readings for resource id %s: %s",resource.id,len(filtered_readings))
        return filtered_readings
    except requests.Timeout as ex:
        _LOGGER.error("Timeout: %s", ex)
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error("Cannot connect: %s", ex)
    except Exception as ex:  # pylint: disable=broad-except
        if "Request failed" in str(ex):
            _LOGGER.warning(
                "Non-200 Status Code. The Glow API may be experiencing issues"
            )
        else:
            _LOGGER.exception("Unexpected exception: %s. Please open an issue", ex)
    return None


async def tariff_data(hass: HomeAssistant, resource) -> float:
    """Get tariff data from the API."""
    try:
        tariff = await hass.async_add_executor_job(resource.get_tariff)
        _LOGGER.debug(
            "Successful GET to https://api.glowmarkt.com/api/v0-1/resource/%s/tariff",
            resource.id,
        )
        return tariff
    except UnboundLocalError:
        supply = supply_type(resource)
        _LOGGER.warning(
            "No tariff data found for %s meter (id: %s). If you don't see tariff data for this meter in the Bright app, please disable the associated rate and standing charge sensors",
            supply,
            resource.id,
        )
    except requests.Timeout as ex:
        _LOGGER.error("Timeout: %s", ex)
    except requests.exceptions.ConnectionError as ex:
        _LOGGER.error("Cannot connect: %s", ex)
    except Exception as ex:  # pylint: disable=broad-except
        if "Request failed" in str(ex):
            _LOGGER.warning(
                "Non-200 Status Code. The Glow API may be experiencing issues"
            )
        else:
            _LOGGER.exception("Unexpected exception: %s. Please open an issue", ex)
    return None


class HistoricalSensorMixin(PollUpdateMixin, HistoricalSensor, SensorEntity):
    @property
    def statistic_id(self) -> str:
        return self.entity_id

    def get_statistic_metadata(self) -> StatisticMetaData:
        meta = super().get_statistic_metadata()
        meta["has_sum"] = True
        return meta

    async def async_calculate_statistic_data(
        self, hist_states: list[HistoricalState], *, latest: dict | None = None
    ) -> list[StatisticData]:
        accumulated = latest["sum"] if latest else 0

        def hour_block_for_hist_state(hist_state: HistoricalState) -> datetime:
            if hist_state.dt.minute == 0 and hist_state.dt.second == 0:
                dt = hist_state.dt - timedelta(hours=1)
                return dt.replace(minute=0, second=0, microsecond=0)
            else:
                return hist_state.dt.replace(minute=0, second=0, microsecond=0)

        ret = []
        for dt, collection_it in itertools.groupby(
            hist_states, key=hour_block_for_hist_state
        ):
            collection = list(collection_it)
            mean = statistics.mean([x.state for x in collection])
            partial_sum = sum([x.state for x in collection])
            accumulated += partial_sum

            ret.append(
                StatisticData(
                    start=dt,
                    state=partial_sum,
                    mean=mean,
                    sum=accumulated,
                )
            )
        return ret


class HistoricalUsage(HistoricalSensorMixin):
    """Historical sensor object for daily usage."""

    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_has_entity_name = True
    _attr_name = "Historical Usage"
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR

    def __init__(self, hass: HomeAssistant, resource, virtual_entity) -> None:
        """Initialize the sensor."""
        self._attr_unique_id = f"{resource.id}-historical-usage"
        self.hass = hass
        self.initialised = False
        self.resource = resource
        self.virtual_entity = virtual_entity
        self.UPDATE_INTERVAL = SCAN_INTERVAL

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.resource.id)},
            manufacturer="Hildebrand",
            model="Glow (DCC)",
            name=device_name(self.resource, self.virtual_entity),
        )

    async def async_update_historical(self) -> None:
        """Fetch new data for the sensor."""
        if not self.initialised or await should_update():
            t_from = None
            if not self.initialised:
                t_from = await self.hass.async_add_executor_job(
                    self.resource.round, datetime.now() - timedelta(days=30), "P1D"
                )
            readings = await daily_data(self.hass, self.resource, t_from)
            self.initialised = True
            hist_states = []
            for reading in readings:
                hist_states.append(
                    HistoricalState(  # noqa: PERF401
                        state=reading[1].value,
                        dt=dt_util.as_local(reading[0] + timedelta(minutes=1)),
                    )
                )
            self._attr_historical_states = hist_states


class HistoricalCost(HistoricalSensorMixin):
    """Historical sensor object for daily cost."""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_has_entity_name = True
    _attr_name = "Historical Cost"
    _attr_native_unit_of_measurement = "GBP"

    def __init__(self, hass: HomeAssistant, resource, virtual_entity) -> None:
        """Initialize the sensor."""
        self._attr_unique_id = f"{resource.id}-historical-cost"
        self.hass = hass
        self.initialised = False
        self.meter = None
        self.resource = resource
        self.virtual_entity = virtual_entity
        self.UPDATE_INTERVAL = SCAN_INTERVAL

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.meter.resource.id)},
            manufacturer="Hildebrand",
            model="Glow (DCC)",
            name=device_name(self.resource, self.virtual_entity),
        )

    async def async_update_historical(self) -> None:
        """Fetch new data for the sensor."""
        # Get data on initial startup
        if not self.initialised or await should_update():
            t_from = None
            if not self.initialised:
                t_from = await self.hass.async_add_executor_job(
                    self.resource.round, datetime.now() - timedelta(days=30), "P1D"
                )
            readings = await daily_data(self.hass, self.resource, t_from)
            self.initialised = True
            hist_states = []
            for reading in readings:
                hist_states.append(
                    HistoricalState(  # noqa: PERF401
                        state=reading[1].value / 100,
                        dt=dt_util.as_local(reading[0] + timedelta(minutes=1)),
                    )
                )
            self._attr_historical_states = hist_states


class TariffCoordinator(DataUpdateCoordinator):
    """Data update coordinator for the tariff sensors."""

    def __init__(self, hass: HomeAssistant, resource) -> None:
        """Initialize tariff coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name="tariff",
            update_interval=SCAN_INTERVAL,
        )

        self.rate_initialised = False
        self.standing_initialised = False
        self.resource = resource

    async def _async_update_data(self):
        """Fetch data from tariff API endpoint."""
        if not self.standing_initialised or not self.rate_initialised:
            self.standing_initialised = True
            self.rate_initialised = True
            return await tariff_data(self.hass, self.resource)
        if await should_update():
            return await tariff_data(self.hass, self.resource)


class Standing(CoordinatorEntity, SensorEntity):
    """An entity using CoordinatorEntity."""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_has_entity_name = True
    _attr_name = "Standing charge"
    _attr_native_unit_of_measurement = "GBP"
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator, resource, virtual_entity) -> None:
        """Pass coordinator to CoordinatorEntity."""
        super().__init__(coordinator)

        self._attr_unique_id = resource.id + "-tariff"

        self.resource = resource
        self.virtual_entity = virtual_entity

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
            identifiers={(DOMAIN, self.resource.id)},
            manufacturer="Hildebrand",
            model="Glow (DCC)",
            name=device_name(self.resource, self.virtual_entity),
        )


class Rate(CoordinatorEntity, SensorEntity):
    """An entity using CoordinatorEntity."""

    _attr_device_class = None
    _attr_has_entity_name = True
    _attr_icon = "mdi:cash-multiple"
    _attr_name = "Rate"
    _attr_native_unit_of_measurement = "GBP/kWh"
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator, resource, virtual_entity) -> None:
        """Pass coordinator to CoordinatorEntity."""
        super().__init__(coordinator)

        self._attr_unique_id = resource.id + "-rate"

        self.resource = resource
        self.virtual_entity = virtual_entity

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
            identifiers={(DOMAIN, self.resource.id)},
            manufacturer="Hildebrand",
            model="Glow (DCC)",
            name=device_name(self.resource, self.virtual_entity),
        )
