from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
import json
from math import isclose
from pathlib import Path
import struct
import time
from typing import Protocol

from autopatcher.adapters.interfaces import ManipulatorAdapter, ManipulatorLimits
from autopatcher.hardware_safety import LiveHardwareAuthorization, LiveHardwareAuthorizationError
from autopatcher.models.domain import Vec3


CR = b"\r"
ETX = b"\x03"
PURGE_RX_TX = 3
# The showcase bundles no driver; callers configure their own licensed location.
DEFAULT_D2XX_DLL_PATH = Path("DRIVER_DIRECTORY") / "ftd2xx.dll"


class SutterManipulatorError(RuntimeError):
    pass


class SutterTransport(Protocol):
    def write(self, payload: bytes) -> None: ...

    def read(self, size: int) -> bytes: ...

    def close(self) -> None: ...

    def reset_input_buffer(self) -> None: ...

    def reset_output_buffer(self) -> None: ...


class _D2XXLibrary:
    def __init__(self, dll_path: Path) -> None:
        if struct.calcsize("P") != 4:
            raise SutterManipulatorError("The Sutter D2XX backend requires 32-bit Python.")
        if not dll_path.exists():
            raise SutterManipulatorError(f"Official Sutter D2XX DLL not found at '{dll_path}'.")

        self.dll = ctypes.WinDLL(str(dll_path))
        self._configure()

    def _configure(self) -> None:
        self.dll.FT_Open.argtypes = [wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
        self.dll.FT_Open.restype = wintypes.DWORD

        self.dll.FT_Close.argtypes = [ctypes.c_void_p]
        self.dll.FT_Close.restype = wintypes.DWORD

        self.dll.FT_SetTimeouts.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD]
        self.dll.FT_SetTimeouts.restype = wintypes.DWORD

        self.dll.FT_Purge.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        self.dll.FT_Purge.restype = wintypes.DWORD

        self.dll.FT_Write.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.dll.FT_Write.restype = wintypes.DWORD

        self.dll.FT_Read.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.dll.FT_Read.restype = wintypes.DWORD

        self.dll.FT_GetQueueStatus.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        self.dll.FT_GetQueueStatus.restype = wintypes.DWORD


class D2XXTransport:
    def __init__(
        self,
        device_index: int,
        timeout_seconds: float,
        live_authorization: LiveHardwareAuthorization,
        dll_path: Path | None = None,
        library: _D2XXLibrary | None = None,
    ) -> None:
        if not live_authorization.operator_confirmed:
            raise LiveHardwareAuthorizationError("D2XX transport requires live hardware authorization")
        self._timeout_seconds = timeout_seconds
        self._library = library or _D2XXLibrary(dll_path or DEFAULT_D2XX_DLL_PATH)
        self._handle = ctypes.c_void_p()
        self._buffer = bytearray()
        status = int(self._library.dll.FT_Open(device_index, ctypes.byref(self._handle)))
        if status != 0:
            raise SutterManipulatorError(f"FT_Open failed with status {status}")
        self._closed = False
        timeout_ms = max(1, int(round(timeout_seconds * 1000.0)))
        self._check(self._library.dll.FT_SetTimeouts(self._handle, timeout_ms, timeout_ms), "FT_SetTimeouts")

    def write(self, payload: bytes) -> None:
        if not payload:
            return
        written = wintypes.DWORD()
        buffer = ctypes.create_string_buffer(payload)
        status = int(self._library.dll.FT_Write(self._handle, buffer, len(payload), ctypes.byref(written)))
        if status != 0 or written.value != len(payload):
            raise SutterManipulatorError(
                f"FT_Write failed with status {status}, wrote {written.value}/{len(payload)} bytes"
            )

    def read(self, size: int) -> bytes:
        if size <= 0:
            return b""
        deadline = time.monotonic() + self._timeout_seconds
        while len(self._buffer) < size and time.monotonic() < deadline:
            self._buffer.extend(self._read_available())
            if len(self._buffer) >= size:
                break
            time.sleep(0.01)
        chunk = bytes(self._buffer[:size])
        del self._buffer[: len(chunk)]
        return chunk

    def close(self) -> None:
        if not self._closed:
            self._library.dll.FT_Close(self._handle)
            self._closed = True

    def reset_input_buffer(self) -> None:
        self._buffer.clear()
        self._purge()

    def reset_output_buffer(self) -> None:
        self._purge()

    def _purge(self) -> None:
        self._check(self._library.dll.FT_Purge(self._handle, PURGE_RX_TX), "FT_Purge")

    def _read_available(self) -> bytes:
        queued = wintypes.DWORD()
        self._check(self._library.dll.FT_GetQueueStatus(self._handle, ctypes.byref(queued)), "FT_GetQueueStatus")
        if queued.value == 0:
            return b""
        buffer = ctypes.create_string_buffer(queued.value)
        returned = wintypes.DWORD()
        self._check(self._library.dll.FT_Read(self._handle, buffer, queued.value, ctypes.byref(returned)), "FT_Read")
        return buffer.raw[: returned.value]

    @staticmethod
    def _check(status: int, label: str) -> None:
        if int(status) != 0:
            raise SutterManipulatorError(f"{label} failed with status {int(status)}")


@dataclass(slots=True)
class SutterManipulatorProtocol:
    """Official Sutter manipulator USB VCP protocol profile.

    This follows the controller-side command syntax documented by Sutter for
    the current supported controller firmware family in USB VCP mode.
    """

    active_device: int = 1
    um_per_microstep: float = 0.0625
    position_poll_attempts: int = 1
    command_retry_attempts: int = 3
    command_retry_delay_seconds: float = 0.1
    move_reply_timeout_seconds: float = 10.0


@dataclass(slots=True)
class SutterManipulatorConfig:
    port: str | None
    limits_um: ManipulatorLimits
    protocol: SutterManipulatorProtocol = field(default_factory=SutterManipulatorProtocol)
    baudrate: int = 128000
    timeout_seconds: float = 1.0
    transport_backend: str = "serial"
    d2xx_device_index: int = 0
    d2xx_dll_path: Path | None = None


def load_protocol_from_json(path: Path) -> SutterManipulatorProtocol:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SutterManipulatorProtocol(
        active_device=int(payload.get("active_device", 1)),
        um_per_microstep=float(payload.get("um_per_microstep", 0.0625)),
        position_poll_attempts=int(payload.get("position_poll_attempts", 1)),
        command_retry_attempts=int(payload.get("command_retry_attempts", 3)),
        command_retry_delay_seconds=float(payload.get("command_retry_delay_seconds", 0.1)),
        move_reply_timeout_seconds=float(payload.get("move_reply_timeout_seconds", 10.0)),
    )


class PySerialTransport:
    def __init__(
        self,
        port: str,
        baudrate: int,
        timeout_seconds: float,
        *,
        live_authorization: LiveHardwareAuthorization,
    ) -> None:
        if not live_authorization.operator_confirmed:
            raise LiveHardwareAuthorizationError("serial transport requires live hardware authorization")
        try:
            import serial
        except ImportError as exc:
            raise SutterManipulatorError(
                "pyserial is required for live Sutter manipulator communication. Install it with `pip install pyserial`."
            ) from exc

        self._serial = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=timeout_seconds,
            bytesize=8,
            parity="N",
            stopbits=1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

    def write(self, payload: bytes) -> None:
        self._serial.write(payload)

    def read(self, size: int) -> bytes:
        return self._serial.read(size)

    def close(self) -> None:
        self._serial.close()

    def reset_input_buffer(self) -> None:
        self._serial.reset_input_buffer()

    def reset_output_buffer(self) -> None:
        self._serial.reset_output_buffer()


class SutterManipulatorAdapter(ManipulatorAdapter):
    """Official Sutter manipulator adapter for USB VCP mode.

    Implemented from Sutter's published USB VCP settings and the
    external-control command syntax used by the current manipulator system.
    """

    def __init__(
        self,
        config: SutterManipulatorConfig,
        transport: SutterTransport | None = None,
        live_authorization: LiveHardwareAuthorization | None = None,
    ) -> None:
        if config.protocol.um_per_microstep <= 0:
            raise ValueError("um_per_microstep must be positive")
        if config.protocol.position_poll_attempts < 1:
            raise ValueError("position_poll_attempts must be at least 1")
        if config.protocol.command_retry_attempts < 1:
            raise ValueError("command_retry_attempts must be at least 1")
        if config.protocol.command_retry_delay_seconds < 0:
            raise ValueError("command_retry_delay_seconds must be non-negative")
        if config.protocol.move_reply_timeout_seconds <= 0:
            raise ValueError("move_reply_timeout_seconds must be positive")
        if not 1 <= config.protocol.active_device <= 4:
            raise ValueError("active_device must be between 1 and 4")
        if config.transport_backend not in {"serial", "d2xx"}:
            raise ValueError("transport_backend must be 'serial' or 'd2xx'")
        if config.d2xx_device_index < 0:
            raise ValueError("d2xx_device_index must be non-negative")

        if transport is None and live_authorization is None:
            raise LiveHardwareAuthorizationError(
                "opening a live Sutter transport requires explicit live hardware authorization"
            )

        self.config = config
        if transport is None:
            assert live_authorization is not None
            self.transport = self._build_transport(config, live_authorization)
        else:
            self.transport = transport
        self._ensure_active_device_selected()

    def close(self) -> None:
        self.transport.close()

    def __enter__(self) -> SutterManipulatorAdapter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_position(self) -> Vec3:
        def _read_position() -> Vec3:
            self.transport.write(b"C")
            payload = self._read_exactly(14)
            device, x_steps, y_steps, z_steps, terminator = self._parse_position_payload(payload)
            if terminator != 13:
                raise SutterManipulatorError(f"Unexpected position terminator byte: {terminator!r}")
            if device != self.config.protocol.active_device:
                raise SutterManipulatorError(
                    f"Controller responded for active device {device}, expected {self.config.protocol.active_device}"
                )
            return self._steps_to_vec3((x_steps, y_steps, z_steps))

        return self._with_command_retries(_read_position)

    def move_absolute(self, target: Vec3) -> None:
        self._assert_within_limits(target)
        target_steps = self._vec3_to_steps(target)
        self._with_command_retries(lambda: self._write_move_and_expect_reply(target_steps))
        self._settle_to_expected(target)

    def move_relative(self, delta: Vec3) -> None:
        origin = self.get_position()
        target = Vec3(origin.x_um + delta.x_um, origin.y_um + delta.y_um, origin.z_um + delta.z_um)
        self.move_absolute(target)

    def move_absolute_straight_line(self, target: Vec3, speed_level: int) -> None:
        self._assert_within_limits(target)
        self._assert_valid_speed_level(speed_level)
        target_steps = self._vec3_to_steps(target)
        self._with_command_retries(lambda: self._write_straight_line_move_and_expect_reply(target_steps, speed_level))
        self._settle_to_expected(target)

    def move_relative_straight_line(self, delta: Vec3, speed_level: int) -> None:
        origin = self.get_position()
        target = Vec3(origin.x_um + delta.x_um, origin.y_um + delta.y_um, origin.z_um + delta.z_um)
        self.move_absolute_straight_line(target, speed_level)

    def get_limits(self) -> ManipulatorLimits:
        return self.config.limits_um

    def stop(self) -> None:
        self._with_command_retries(self._write_stop_and_expect_reply)

    def set_mode(self, mode: int) -> None:
        if not 0 <= mode <= 9:
            raise ValueError("mode must be between 0 and 9")
        self._with_command_retries(lambda: self._write_mode_and_expect_reply(mode))

    def home(self) -> None:
        self._with_command_retries(self._write_home_and_expect_reply)

    def move_to_work_position(self) -> None:
        self._with_command_retries(self._write_work_position_and_expect_reply)

    def calibrate(self) -> None:
        self._with_command_retries(self._write_calibrate_and_expect_reply)

    def _select_active_device(self, device: int) -> None:
        self.transport.write(b"I" + bytes([device]))
        reply = self._read_exactly(2)
        if reply == b"E\r":
            raise SutterManipulatorError(f"Controller rejected active device selection: {device}")
        if reply != bytes([device]) + CR:
            raise SutterManipulatorError(f"Unexpected device-select reply: {reply!r}")

    def _ensure_active_device_selected(self) -> None:
        last_error: SutterManipulatorError | None = None
        for attempt_index in range(self.config.protocol.command_retry_attempts):
            try:
                self._reset_transport_buffers()
                self._select_active_device(self.config.protocol.active_device)
                return
            except SutterManipulatorError as exc:
                last_error = exc
                if attempt_index + 1 >= self.config.protocol.command_retry_attempts:
                    break
                self._pause_before_retry()
        raise SutterManipulatorError(
            f"Failed to select active device {self.config.protocol.active_device} after "
            f"{self.config.protocol.command_retry_attempts} attempt(s): {last_error}"
        ) from last_error

    def _expect_single_byte_reply(self, expected: bytes, timeout_seconds: float | None = None) -> None:
        reply = self._read_exactly(len(expected), timeout_seconds=timeout_seconds)
        if reply != expected:
            raise SutterManipulatorError(f"Unexpected controller reply: {reply!r}, expected {expected!r}")

    def _settle_to_expected(self, expected: Vec3) -> None:
        last_actual: Vec3 | None = None
        for _ in range(self.config.protocol.position_poll_attempts):
            last_actual = self.get_position()
            if self._positions_match(last_actual, expected):
                return
        raise SutterManipulatorError(f"Manipulator reported {last_actual} after move, expected {expected}")

    def _write_move_and_expect_reply(self, target_steps: tuple[int, int, int]) -> None:
        self.transport.write(b"M" + struct.pack("<iii", *target_steps))
        self._expect_single_byte_reply(CR, timeout_seconds=self.config.protocol.move_reply_timeout_seconds)

    def _write_straight_line_move_and_expect_reply(self, target_steps: tuple[int, int, int], speed_level: int) -> None:
        self.transport.write(b"F")
        self._expect_single_byte_reply(CR)
        self.transport.write(b"S" + bytes([speed_level]))
        time.sleep(0.03)
        self.transport.write(struct.pack("<iii", *target_steps))
        self._expect_single_byte_reply(CR, timeout_seconds=self.config.protocol.move_reply_timeout_seconds)

    def _write_stop_and_expect_reply(self) -> None:
        self.transport.write(ETX)
        self._expect_single_byte_reply(CR)

    def _write_mode_and_expect_reply(self, mode: int) -> None:
        self.transport.write(b"L" + bytes([mode]))
        self._expect_single_byte_reply(CR)

    def _write_home_and_expect_reply(self) -> None:
        self.transport.write(b"H")
        self._expect_single_byte_reply(CR, timeout_seconds=self.config.protocol.move_reply_timeout_seconds)

    def _write_work_position_and_expect_reply(self) -> None:
        self.transport.write(b"Y")
        self._expect_single_byte_reply(CR, timeout_seconds=self.config.protocol.move_reply_timeout_seconds)

    def _write_calibrate_and_expect_reply(self) -> None:
        self.transport.write(b"N")
        self._expect_single_byte_reply(CR, timeout_seconds=self.config.protocol.move_reply_timeout_seconds)

    def _with_command_retries(self, operation):
        last_error: SutterManipulatorError | None = None
        for attempt_index in range(self.config.protocol.command_retry_attempts):
            try:
                return operation()
            except SutterManipulatorError as exc:
                last_error = exc
                if attempt_index + 1 >= self.config.protocol.command_retry_attempts:
                    break
                self._pause_before_retry()
                self._reset_transport_buffers()
                self._select_active_device(self.config.protocol.active_device)
        raise SutterManipulatorError(
            f"Controller command failed after {self.config.protocol.command_retry_attempts} attempt(s): {last_error}"
        ) from last_error

    def _pause_before_retry(self) -> None:
        if self.config.protocol.command_retry_delay_seconds > 0:
            time.sleep(self.config.protocol.command_retry_delay_seconds)

    def _reset_transport_buffers(self) -> None:
        reset_input = getattr(self.transport, "reset_input_buffer", None)
        if callable(reset_input):
            reset_input()
        reset_output = getattr(self.transport, "reset_output_buffer", None)
        if callable(reset_output):
            reset_output()

    @staticmethod
    def _build_transport(
        config: SutterManipulatorConfig,
        live_authorization: LiveHardwareAuthorization,
    ) -> SutterTransport:
        if config.transport_backend == "d2xx":
            return D2XXTransport(
                device_index=config.d2xx_device_index,
                timeout_seconds=config.timeout_seconds,
                live_authorization=live_authorization,
                dll_path=config.d2xx_dll_path,
            )
        if not config.port:
            raise ValueError("port is required when transport_backend='serial'")
        return PySerialTransport(
            config.port,
            config.baudrate,
            config.timeout_seconds,
            live_authorization=live_authorization,
        )

    def _assert_within_limits(self, target: Vec3) -> None:
        minimum = self.config.limits_um.minimum
        maximum = self.config.limits_um.maximum
        if not (minimum.x_um <= target.x_um <= maximum.x_um):
            raise SutterManipulatorError(f"Target x {target.x_um} um is outside limits [{minimum.x_um}, {maximum.x_um}]")
        if not (minimum.y_um <= target.y_um <= maximum.y_um):
            raise SutterManipulatorError(f"Target y {target.y_um} um is outside limits [{minimum.y_um}, {maximum.y_um}]")
        if not (minimum.z_um <= target.z_um <= maximum.z_um):
            raise SutterManipulatorError(f"Target z {target.z_um} um is outside limits [{minimum.z_um}, {maximum.z_um}]")

    @staticmethod
    def _assert_valid_speed_level(speed_level: int) -> None:
        if not 0 <= speed_level <= 15:
            raise ValueError("speed_level must be between 0 and 15")

    def _positions_match(self, actual: Vec3, expected: Vec3) -> bool:
        tolerance_um = self.config.protocol.um_per_microstep
        return (
            isclose(actual.x_um, expected.x_um, abs_tol=tolerance_um)
            and isclose(actual.y_um, expected.y_um, abs_tol=tolerance_um)
            and isclose(actual.z_um, expected.z_um, abs_tol=tolerance_um)
        )

    def _vec3_to_steps(self, value: Vec3) -> tuple[int, int, int]:
        return (
            self._um_to_steps(value.x_um),
            self._um_to_steps(value.y_um),
            self._um_to_steps(value.z_um),
        )

    def _steps_to_vec3(self, steps: tuple[int, int, int]) -> Vec3:
        return Vec3(
            x_um=self._steps_to_um(steps[0]),
            y_um=self._steps_to_um(steps[1]),
            z_um=self._steps_to_um(steps[2]),
        )

    def _um_to_steps(self, value_um: float) -> int:
        return int(round(value_um / self.config.protocol.um_per_microstep))

    def _steps_to_um(self, value_steps: int) -> float:
        return value_steps * self.config.protocol.um_per_microstep

    @staticmethod
    def _parse_position_payload(payload: bytes) -> tuple[int, int, int, int, int]:
        if len(payload) != 14:
            raise SutterManipulatorError(f"Expected 14-byte position payload, got {len(payload)} bytes")
        return struct.unpack("<BiiiB", payload)

    def _read_exactly(self, size: int, timeout_seconds: float | None = None) -> bytes:
        deadline = time.monotonic() + (timeout_seconds if timeout_seconds is not None else self.config.timeout_seconds)
        payload = b""
        while len(payload) < size:
            chunk = self.transport.read(size - len(payload))
            if chunk:
                payload += chunk
                continue
            if time.monotonic() >= deadline:
                break
        if len(payload) != size:
            raise SutterManipulatorError(f"Expected {size} bytes from controller, received {len(payload)} bytes")
        return payload
