from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class ScaleCalibrator:
    AUTO_PREFIX = '__auto_profile__'

    def __init__(
        self,
        cache_path: str,
        default_mm_per_px: float,
        board_size: tuple[int, int] = (7, 7),
        board_size_candidates: list[tuple[int, int]] | None = None,
        square_size_mm: float = 5.0,
        charuco_enabled: bool = False,
        charuco_squares_x: int = 5,
        charuco_squares_y: int = 7,
        charuco_square_size_mm: float = 8.0,
        charuco_marker_size_mm: float = 6.0,
        charuco_dictionary: str = 'DICT_4X4_50',
        auto_profile_enabled: bool = True,
        auto_min_samples: int = 3,
        auto_stable_samples: int = 8,
        auto_max_cv: float = 0.35,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.default_mm_per_px = float(default_mm_per_px)
        self.board_size = tuple(board_size)
        self.board_size_candidates = self._normalize_board_size_candidates(
            self.board_size,
            board_size_candidates,
        )
        self.square_size_mm = float(square_size_mm)

        self.charuco_enabled = bool(charuco_enabled)
        self.charuco_squares_x = int(charuco_squares_x)
        self.charuco_squares_y = int(charuco_squares_y)
        self.charuco_square_size_mm = float(charuco_square_size_mm)
        self.charuco_marker_size_mm = float(charuco_marker_size_mm)
        self.charuco_dictionary = str(charuco_dictionary)
        self.auto_profile_enabled = bool(auto_profile_enabled)
        self.auto_min_samples = max(1, int(auto_min_samples))
        self.auto_stable_samples = max(self.auto_min_samples, int(auto_stable_samples))
        self.auto_max_cv = max(0.01, float(auto_max_cv))

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding='utf-8'))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    def _save_cache(self) -> None:
        self.cache_path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=True),
            encoding='utf-8',
        )

    @staticmethod
    def _image_fingerprint(image: np.ndarray) -> str:
        return hashlib.sha1(image.tobytes()).hexdigest()[:12]

    @staticmethod
    def _normalize_token(value: str) -> str:
        token = ''.join(ch if ch.isalnum() else '_' for ch in str(value).strip().lower())
        token = token.strip('_')
        return token or 'unknown'

    def _auto_profile_keys(self, camera_id: str, source_type: str, crop: str) -> list[str]:
        cam = self._normalize_token(camera_id)
        src = self._normalize_token(source_type)
        crp = self._normalize_token(crop)
        return [
            f'{self.AUTO_PREFIX}:cam:{src}:{cam}:{crp}',
            f'{self.AUTO_PREFIX}:source:{src}:{crp}',
            f'{self.AUTO_PREFIX}:crop:{crp}',
            f'{self.AUTO_PREFIX}:global',
        ]

    @staticmethod
    def _entry_mean(entry: dict) -> Optional[float]:
        try:
            if 'mean_mm_per_px' in entry:
                return float(entry['mean_mm_per_px'])
            if 'mm_per_px' in entry:
                return float(entry['mm_per_px'])
        except Exception:
            return None
        return None

    @staticmethod
    def _entry_count(entry: dict) -> int:
        try:
            c = int(entry.get('count', 1))
            return max(1, c)
        except Exception:
            return 1

    @staticmethod
    def _entry_cv(entry: dict) -> Optional[float]:
        mean_val = ScaleCalibrator._entry_mean(entry)
        if mean_val is None or mean_val <= 0:
            return None
        count = ScaleCalibrator._entry_count(entry)
        if count < 2:
            return None
        try:
            m2 = float(entry.get('m2', 0.0))
            if m2 < 0:
                m2 = 0.0
            var = m2 / float(max(1, count - 1))
            std = math.sqrt(var)
            return std / mean_val
        except Exception:
            return None

    def _update_running_stats(self, key: str, value_mm_per_px: float) -> None:
        entry = self._cache.get(key, {})
        mean_prev = self._entry_mean(entry)
        if mean_prev is None:
            mean_prev = float(value_mm_per_px)
        try:
            count_prev = int(entry.get('count', 0))
        except Exception:
            count_prev = 0
        if count_prev <= 0 and ('mean_mm_per_px' in entry or 'mm_per_px' in entry):
            count_prev = 1
        m2_prev = float(entry.get('m2', 0.0))

        count_new = count_prev + 1
        delta = float(value_mm_per_px) - mean_prev
        mean_new = mean_prev + delta / float(count_new)
        delta2 = float(value_mm_per_px) - mean_new
        m2_new = max(0.0, m2_prev + delta * delta2)

        self._cache[key] = {
            'mean_mm_per_px': float(mean_new),
            'mm_per_px': float(mean_new),
            'count': int(count_new),
            'm2': float(m2_new),
        }

    def update_auto_scale(
        self,
        mm_per_px: float,
        camera_id: str = 'default',
        source_type: str = 'unknown',
        crop: str = 'Unknown',
    ) -> None:
        if not self.auto_profile_enabled:
            return
        try:
            val = float(mm_per_px)
        except Exception:
            return
        if (not math.isfinite(val)) or val <= 0.0:
            return
        # Conservative bounds to avoid poisoning profile by outliers.
        if not (0.01 <= val <= 1.0):
            return

        for key in self._auto_profile_keys(camera_id=camera_id, source_type=source_type, crop=crop):
            self._update_running_stats(key, val)
        self._save_cache()

    def get_auto_scale(
        self,
        camera_id: str = 'default',
        source_type: str = 'unknown',
        crop: str = 'Unknown',
        min_samples: Optional[int] = None,
    ) -> tuple[Optional[float], Optional[str]]:
        if not self.auto_profile_enabled:
            return None, None
        required_samples = self.auto_min_samples if min_samples is None else max(1, int(min_samples))
        for key in self._auto_profile_keys(camera_id=camera_id, source_type=source_type, crop=crop):
            entry = self._cache.get(key)
            if not isinstance(entry, dict):
                continue
            mean_val = self._entry_mean(entry)
            if mean_val is None or mean_val <= 0.0 or (not math.isfinite(mean_val)):
                continue
            count = self._entry_count(entry)
            if count < required_samples:
                continue
            return float(mean_val), key
        return None, None

    def is_auto_profile_stable(
        self,
        camera_id: str = 'default',
        source_type: str = 'unknown',
        crop: str = 'Unknown',
    ) -> bool:
        if not self.auto_profile_enabled:
            return False
        for key in self._auto_profile_keys(camera_id=camera_id, source_type=source_type, crop=crop):
            entry = self._cache.get(key)
            if not isinstance(entry, dict):
                continue
            count = self._entry_count(entry)
            if count < self.auto_stable_samples:
                continue
            mean_val = self._entry_mean(entry)
            if mean_val is None or mean_val <= 0.0 or (not math.isfinite(mean_val)):
                continue
            cv = self._entry_cv(entry)
            if cv is None:
                return True
            if cv <= self.auto_max_cv:
                return True
        return False

    def _resolve_aruco_dictionary(self):
        if not hasattr(cv2, 'aruco'):
            return None
        dict_id = getattr(cv2.aruco, self.charuco_dictionary, None)
        if dict_id is None:
            dict_id = getattr(cv2.aruco, 'DICT_4X4_50', None)
        if dict_id is None:
            return None
        return cv2.aruco.getPredefinedDictionary(dict_id)

    @staticmethod
    def _normalize_board_size_candidates(
        primary: tuple[int, int],
        candidates: list[tuple[int, int]] | None,
    ) -> list[tuple[int, int]]:
        ordered: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def _push(pair: tuple[int, int]) -> None:
            try:
                cols = int(pair[0])
                rows = int(pair[1])
            except Exception:
                return
            if cols < 2 or rows < 2:
                return
            key = (cols, rows)
            if key in seen:
                return
            seen.add(key)
            ordered.append(key)

        _push(primary)
        _push((primary[1], primary[0]))
        for cand in candidates or []:
            if not isinstance(cand, (list, tuple)) or len(cand) != 2:
                continue
            _push((cand[0], cand[1]))
            _push((cand[1], cand[0]))

        return ordered or [(7, 7)]

    @staticmethod
    def _robust_mm_per_px(distances_px: list[float], square_mm: float) -> Optional[float]:
        if not distances_px:
            return None

        arr = np.asarray(distances_px, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        arr = arr[arr > 0.0]
        if arr.size == 0:
            return None

        q1, q3 = np.percentile(arr, [25, 75])
        iqr = max(1e-6, q3 - q1)
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        filtered = arr[(arr >= lo) & (arr <= hi)]
        if filtered.size == 0:
            filtered = arr

        px_per_square = float(np.median(filtered))
        if px_per_square <= 0:
            return None
        return float(square_mm / px_per_square)

    @staticmethod
    def _scale_from_corner_grid(
        corners: np.ndarray,
        board_size: tuple[int, int],
        square_size_mm: float,
    ) -> Optional[float]:
        corners = corners.reshape(-1, 2).astype(np.float32)
        cols, rows = int(board_size[0]), int(board_size[1])
        if len(corners) != cols * rows:
            return None

        horizontal_dist = []
        vertical_dist = []

        for r in range(rows):
            for c in range(cols - 1):
                i1 = r * cols + c
                i2 = i1 + 1
                horizontal_dist.append(float(np.linalg.norm(corners[i1] - corners[i2])))

        for c in range(cols):
            for r in range(rows - 1):
                i1 = r * cols + c
                i2 = (r + 1) * cols + c
                vertical_dist.append(float(np.linalg.norm(corners[i1] - corners[i2])))

        return ScaleCalibrator._robust_mm_per_px(horizontal_dist + vertical_dist, float(square_size_mm))

    @staticmethod
    def _find_chessboard_corners(gray: np.ndarray, board_size: tuple[int, int]) -> Optional[np.ndarray]:
        def _downscale_if_large(img_gray: np.ndarray, max_dim: int = 1600) -> tuple[np.ndarray, float, float]:
            h, w = img_gray.shape[:2]
            m = max(h, w)
            if m <= max_dim:
                return img_gray, 1.0, 1.0
            scale = float(max_dim) / float(max(1, m))
            nw = max(32, int(round(w * scale)))
            nh = max(32, int(round(h * scale)))
            resized = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_AREA)
            sx = float(w) / float(max(1, nw))
            sy = float(h) / float(max(1, nh))
            return resized, sx, sy

        def _single_pass(img_gray: np.ndarray) -> Optional[np.ndarray]:
            found = False
            corners = None

            # Fast pass first: dramatically cheaper on non-checkerboard plant images.
            flags_fast = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK
            )
            found, corners = cv2.findChessboardCorners(img_gray, board_size, flags_fast)
            if found and corners is not None:
                term = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    50,
                    0.001,
                )
                cv2.cornerSubPix(img_gray, corners, (7, 7), (-1, -1), term)

            # If fast pass failed, try robust SB detector (lighter flags first).
            if ((not found) or corners is None) and hasattr(cv2, 'findChessboardCornersSB'):
                sb_flags = 0
                if hasattr(cv2, 'CALIB_CB_NORMALIZE_IMAGE'):
                    sb_flags |= int(getattr(cv2, 'CALIB_CB_NORMALIZE_IMAGE'))
                try:
                    found, corners = cv2.findChessboardCornersSB(img_gray, board_size, sb_flags)
                except Exception:
                    found, corners = False, None

            # Final fallback: exhaustive SB on downscaled variants only.
            if ((not found) or corners is None) and hasattr(cv2, 'findChessboardCornersSB'):
                sb_flags = 0
                for flag_name in ('CALIB_CB_EXHAUSTIVE', 'CALIB_CB_ACCURACY', 'CALIB_CB_NORMALIZE_IMAGE'):
                    if hasattr(cv2, flag_name):
                        sb_flags |= int(getattr(cv2, flag_name))
                try:
                    found, corners = cv2.findChessboardCornersSB(img_gray, board_size, sb_flags)
                except Exception:
                    found, corners = False, None

            if (not found) or corners is None:
                return None
            return corners

        base, base_sx, base_sy = _downscale_if_large(gray)
        variants: list[np.ndarray] = [base]
        try:
            variants.append(cv2.equalizeHist(base))
        except Exception:
            pass
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            variants.append(clahe.apply(base))
        except Exception:
            pass
        variants.append(cv2.GaussianBlur(base, (3, 3), 0))
        variants.append(cv2.bitwise_not(base))
        variants.append(cv2.GaussianBlur(cv2.bitwise_not(base), (3, 3), 0))
        # High contrast variant helps under dim or low-contrast captures.
        try:
            _, th = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(th)
            variants.append(cv2.bitwise_not(th))
        except Exception:
            pass

        h, w = base.shape[:2]
        if max(h, w) < 1200:
            up = cv2.resize(base, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            variants.append(up)
            variants.append(cv2.equalizeHist(up))

        for img_variant in variants:
            corners = _single_pass(img_variant)
            if corners is None:
                continue
            vh, vw = img_variant.shape[:2]
            if vh != h or vw != w:
                scale_x = float(w) / float(max(1, vw))
                scale_y = float(h) / float(max(1, vh))
                corners = corners.copy()
                corners[:, 0, 0] *= scale_x
                corners[:, 0, 1] *= scale_y
            if (base_sx != 1.0) or (base_sy != 1.0):
                corners = corners.copy()
                corners[:, 0, 0] *= base_sx
                corners[:, 0, 1] *= base_sy
            return corners
        return None

    def _estimate_scale_from_chessboard(self, image: np.ndarray) -> Optional[float]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for board_size in self.board_size_candidates:
            corners = self._find_chessboard_corners(gray, board_size)
            if corners is None:
                continue
            scale = self._scale_from_corner_grid(corners, board_size, self.square_size_mm)
            if scale is None:
                continue
            logger.info(
                'Chessboard scale found with inner corners %sx%s',
                board_size[0],
                board_size[1],
            )
            return float(scale)
        return None

    def _estimate_scale_from_charuco(self, image: np.ndarray) -> Optional[float]:
        if not self.charuco_enabled:
            return None
        if not hasattr(cv2, 'aruco'):
            return None

        dictionary = self._resolve_aruco_dictionary()
        if dictionary is None:
            return None

        try:
            board = cv2.aruco.CharucoBoard(
                (self.charuco_squares_x, self.charuco_squares_y),
                self.charuco_square_size_mm,
                self.charuco_marker_size_mm,
                dictionary,
            )
        except Exception:
            board = cv2.aruco.CharucoBoard_create(
                self.charuco_squares_x,
                self.charuco_squares_y,
                self.charuco_square_size_mm,
                self.charuco_marker_size_mm,
                dictionary,
            )

        if hasattr(board, 'setLegacyPattern'):
            try:
                board.setLegacyPattern(True)
            except Exception:
                pass

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        charuco_corners = None
        charuco_ids = None
        if hasattr(cv2.aruco, 'CharucoDetector'):
            try:
                charuco_detector = cv2.aruco.CharucoDetector(board)
                charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
            except Exception:
                charuco_corners, charuco_ids = None, None
        elif hasattr(cv2.aruco, 'interpolateCornersCharuco'):
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, params)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
            if marker_ids is not None and len(marker_ids) >= 4:
                ok, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=marker_corners,
                    markerIds=marker_ids,
                    image=gray,
                    board=board,
                )
                if ok:
                    charuco_corners, charuco_ids = c_corners, c_ids

        if charuco_corners is None or charuco_ids is None:
            return None
        if len(charuco_ids) < 6:
            return None

        corner_map: dict[int, np.ndarray] = {}
        for c, i in zip(charuco_corners.reshape(-1, 2), charuco_ids.reshape(-1)):
            corner_map[int(i)] = np.asarray(c, dtype=np.float32)

        sx = self.charuco_squares_x - 1
        sy = self.charuco_squares_y - 1
        distances: list[float] = []

        for y in range(sy):
            for x in range(sx):
                idx = y * sx + x
                right = idx + 1 if x + 1 < sx else None
                down = idx + sx if y + 1 < sy else None
                p = corner_map.get(idx)
                if p is None:
                    continue
                if right is not None and right in corner_map:
                    distances.append(float(np.linalg.norm(p - corner_map[right])))
                if down is not None and down in corner_map:
                    distances.append(float(np.linalg.norm(p - corner_map[down])))

        return self._robust_mm_per_px(distances, self.charuco_square_size_mm)

    def upsert_scale(self, camera_id: str, mm_per_px: float, fingerprint: str = 'manual') -> None:
        self._cache[str(camera_id)] = {
            'mm_per_px': float(mm_per_px),
            'fingerprint': str(fingerprint),
            'validated': True,
            'updated_at': self._utc_now_iso(),
            'calibration_source': str(fingerprint).split('_', 1)[0] if str(fingerprint).strip() else 'manual',
        }
        self._save_cache()

    def get_profile(self, camera_id: str) -> dict | None:
        entry = self._cache.get(str(camera_id))
        if not isinstance(entry, dict):
            return None
        mm_per_px = entry.get('mm_per_px', entry.get('mean_mm_per_px'))
        if mm_per_px is None:
            return None
        try:
            mm_per_px = float(mm_per_px)
        except Exception:
            return None
        return {
            'camera_id': str(camera_id),
            'mm_per_px': mm_per_px,
            'validated': bool(entry.get('validated', False)) or self.is_cache_scale_validated(str(camera_id)),
            'fingerprint': str(entry.get('fingerprint', '')),
            'calibration_source': str(entry.get('calibration_source', 'cache')),
            'updated_at': str(entry.get('updated_at', '')),
        }

    def list_profiles(self, validated_only: bool = False) -> list[dict]:
        rows: list[dict] = []
        for camera_id, entry in self._cache.items():
            if not isinstance(entry, dict):
                continue
            if str(camera_id).startswith(self.AUTO_PREFIX):
                continue
            profile = self.get_profile(str(camera_id))
            if profile is None:
                continue
            if validated_only and (not bool(profile.get('validated', False))):
                continue
            rows.append(profile)
        rows.sort(key=lambda x: (x.get('camera_id') != 'default', str(x.get('camera_id', ''))))
        return rows

    def is_cache_scale_validated(self, camera_id: str) -> bool:
        entry = self._cache.get(str(camera_id))
        if not isinstance(entry, dict):
            return False
        if bool(entry.get('validated', False)):
            return True
        fp = str(entry.get('fingerprint', '')).strip().lower()
        # Keep validation conservative: only explicit fit/manual/trusted markers.
        trusted_prefixes = ('fit_', 'manual', 'trusted_', 'calib_', 'chessboard_', 'charuco_', 'source:')
        return any(fp.startswith(prefix) for prefix in trusted_prefixes)

    def calibrate_and_store(
        self,
        image: Optional[np.ndarray],
        camera_id: str,
    ) -> tuple[Optional[float], Optional[str]]:
        scale, source = self.estimate_scale(image)
        if scale is None or source is None:
            return None, None
        fp_raw = self._image_fingerprint(image) if image is not None else 'none'
        fingerprint = f'{source}_{fp_raw}'
        self._cache[str(camera_id)] = {
            'mm_per_px': float(scale),
            'fingerprint': fingerprint,
            'validated': True,
            'updated_at': self._utc_now_iso(),
            'calibration_source': str(source),
        }
        self._save_cache()
        return float(scale), str(source)

    def estimate_scale(self, image: Optional[np.ndarray]) -> tuple[Optional[float], Optional[str]]:
        if image is None:
            return None, None

        scale = self._estimate_scale_from_chessboard(image)
        if scale is not None:
            return scale, 'chessboard'

        scale_charuco = self._estimate_scale_from_charuco(image)
        if scale_charuco is not None:
            return scale_charuco, 'charuco'

        return None, None

    def get_scale(
        self,
        image: Optional[np.ndarray],
        camera_id: str = 'default',
        use_cache: bool = True,
    ) -> tuple[float, str]:
        scale, source = self.calibrate_and_store(image=image, camera_id=camera_id)
        if scale is not None and source is not None:
            return float(scale), str(source)

        if use_cache and camera_id in self._cache:
            return float(self._cache[camera_id]['mm_per_px']), 'cache'

        logger.debug('Calibration fallback used for camera_id=%s', camera_id)
        return self.default_mm_per_px, 'fallback'
