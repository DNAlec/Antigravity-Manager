use serde::{Deserialize, Serialize};

/// Per-instance QoS policy. Durations are seconds; omitted fields retain the
/// established policy so existing configuration files keep their behavior.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "UncheckedQoSConfig")]
pub struct QoSConfig {
    pub normal_threshold_secs: f64,
    pub deep_thinking_threshold_secs: f64,
    pub recovery_threshold_secs: f64,
    pub no_peer_grace_secs: f64,
    pub probe_timeout_secs: f64,
    pub probe_success_threshold_secs: f64,
    /// Six fixed levels keep existing jailed accounts valid during hot updates.
    pub jail_ladder_secs: [u64; 6],
}

impl Default for QoSConfig {
    fn default() -> Self {
        Self {
            normal_threshold_secs: 20.0,
            deep_thinking_threshold_secs: 35.0,
            recovery_threshold_secs: 18.5,
            no_peer_grace_secs: 10.0,
            probe_timeout_secs: 8.0,
            probe_success_threshold_secs: 8.0,
            jail_ladder_secs: [120, 300, 600, 900, 1200, 1800],
        }
    }
}

impl QoSConfig {
    /// Validate direct Rust construction as well as deserialized configuration.
    pub fn validate(&self) -> Result<(), String> {
        for (field, value, min, max) in [
            (
                "normal_threshold_secs",
                self.normal_threshold_secs,
                1.0,
                600.0,
            ),
            (
                "deep_thinking_threshold_secs",
                self.deep_thinking_threshold_secs,
                1.0,
                600.0,
            ),
            (
                "recovery_threshold_secs",
                self.recovery_threshold_secs,
                1.0,
                600.0,
            ),
            ("no_peer_grace_secs", self.no_peer_grace_secs, 0.0, 600.0),
            ("probe_timeout_secs", self.probe_timeout_secs, 0.1, 120.0),
            (
                "probe_success_threshold_secs",
                self.probe_success_threshold_secs,
                0.1,
                120.0,
            ),
        ] {
            if !value.is_finite() || !(min..=max).contains(&value) {
                return Err(format!(
                    "proxy.qos.{field} must be finite and between {min} and {max} seconds"
                ));
            }
        }

        if self.recovery_threshold_secs >= self.normal_threshold_secs {
            return Err(
                "proxy.qos.recovery_threshold_secs must be less than normal_threshold_secs"
                    .to_string(),
            );
        }
        if self.deep_thinking_threshold_secs < self.normal_threshold_secs {
            return Err(
                "proxy.qos.deep_thinking_threshold_secs must be at least normal_threshold_secs"
                    .to_string(),
            );
        }
        if self.probe_success_threshold_secs > self.probe_timeout_secs {
            return Err(
                "proxy.qos.probe_success_threshold_secs must not exceed probe_timeout_secs"
                    .to_string(),
            );
        }

        for (level, &seconds) in self.jail_ladder_secs.iter().enumerate() {
            if !(1..=86400).contains(&seconds) {
                return Err(format!(
                    "proxy.qos.jail_ladder_secs[{level}] must be between 1 and 86400 seconds"
                ));
            }
        }
        if self
            .jail_ladder_secs
            .windows(2)
            .any(|pair| pair[0] > pair[1])
        {
            return Err("proxy.qos.jail_ladder_secs must be nondecreasing".to_string());
        }
        Ok(())
    }
}

// Keep deserialization validation at the type boundary: configuration loaded
// from disk, HTTP, and desktop commands all pass through the same checks.
#[derive(Deserialize)]
#[serde(default)]
struct UncheckedQoSConfig {
    normal_threshold_secs: f64,
    deep_thinking_threshold_secs: f64,
    recovery_threshold_secs: f64,
    no_peer_grace_secs: f64,
    probe_timeout_secs: f64,
    probe_success_threshold_secs: f64,
    jail_ladder_secs: [u64; 6],
}

impl Default for UncheckedQoSConfig {
    fn default() -> Self {
        let defaults = QoSConfig::default();
        Self {
            normal_threshold_secs: defaults.normal_threshold_secs,
            deep_thinking_threshold_secs: defaults.deep_thinking_threshold_secs,
            recovery_threshold_secs: defaults.recovery_threshold_secs,
            no_peer_grace_secs: defaults.no_peer_grace_secs,
            probe_timeout_secs: defaults.probe_timeout_secs,
            probe_success_threshold_secs: defaults.probe_success_threshold_secs,
            jail_ladder_secs: defaults.jail_ladder_secs,
        }
    }
}

impl TryFrom<UncheckedQoSConfig> for QoSConfig {
    type Error = String;

    fn try_from(raw: UncheckedQoSConfig) -> Result<Self, Self::Error> {
        let config = Self {
            normal_threshold_secs: raw.normal_threshold_secs,
            deep_thinking_threshold_secs: raw.deep_thinking_threshold_secs,
            recovery_threshold_secs: raw.recovery_threshold_secs,
            no_peer_grace_secs: raw.no_peer_grace_secs,
            probe_timeout_secs: raw.probe_timeout_secs,
            probe_success_threshold_secs: raw.probe_success_threshold_secs,
            jail_ladder_secs: raw.jail_ladder_secs,
        };
        config.validate()?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_policy_preserves_existing_defaults() {
        let config: QoSConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(
            serde_json::to_value(config).unwrap(),
            json!({
                "normal_threshold_secs": 20.0,
                "deep_thinking_threshold_secs": 35.0,
                "recovery_threshold_secs": 18.5,
                "no_peer_grace_secs": 10.0,
                "probe_timeout_secs": 8.0,
                "probe_success_threshold_secs": 8.0,
                "jail_ladder_secs": [120, 300, 600, 900, 1200, 1800]
            })
        );
    }

    #[test]
    fn partial_policy_preserves_omitted_fields() {
        let config: QoSConfig = serde_json::from_value(json!({
            "no_peer_grace_secs": 0.0,
            "probe_timeout_secs": 12.5
        }))
        .unwrap();
        assert_eq!(
            config,
            QoSConfig {
                no_peer_grace_secs: 0.0,
                probe_timeout_secs: 12.5,
                ..QoSConfig::default()
            }
        );
    }

    #[test]
    fn configured_policy_round_trips_without_losing_fractional_seconds() {
        let expected = QoSConfig {
            normal_threshold_secs: 45.5,
            deep_thinking_threshold_secs: 90.0,
            recovery_threshold_secs: 30.25,
            no_peer_grace_secs: 4.5,
            probe_timeout_secs: 12.5,
            probe_success_threshold_secs: 11.25,
            jail_ladder_secs: [30, 60, 120, 240, 480, 960],
        };
        let encoded = serde_json::to_string(&expected).unwrap();
        assert_eq!(
            serde_json::from_str::<QoSConfig>(&encoded).unwrap(),
            expected
        );
    }

    #[test]
    fn all_duration_fields_reject_nonfinite_and_out_of_range_values() {
        type Accessor = fn(&mut QoSConfig) -> &mut f64;
        let fields: [(&str, Accessor, f64, f64); 6] = [
            (
                "normal_threshold_secs",
                |c| &mut c.normal_threshold_secs,
                1.0,
                600.0,
            ),
            (
                "deep_thinking_threshold_secs",
                |c| &mut c.deep_thinking_threshold_secs,
                1.0,
                600.0,
            ),
            (
                "recovery_threshold_secs",
                |c| &mut c.recovery_threshold_secs,
                1.0,
                600.0,
            ),
            (
                "no_peer_grace_secs",
                |c| &mut c.no_peer_grace_secs,
                0.0,
                600.0,
            ),
            (
                "probe_timeout_secs",
                |c| &mut c.probe_timeout_secs,
                0.1,
                120.0,
            ),
            (
                "probe_success_threshold_secs",
                |c| &mut c.probe_success_threshold_secs,
                0.1,
                120.0,
            ),
        ];
        for (field, access, min, max) in fields {
            for value in [
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                min - 0.01,
                max + 0.01,
            ] {
                let mut config = QoSConfig::default();
                *access(&mut config) = value;
                let error = config.validate().unwrap_err();
                assert!(error.contains(field), "{field}={value}: {error}");
            }
        }
    }

    #[test]
    fn valid_boundary_values_and_equal_ladder_steps_are_accepted() {
        let configs = [
            QoSConfig {
                normal_threshold_secs: 2.0,
                deep_thinking_threshold_secs: 2.0,
                recovery_threshold_secs: 1.0,
                no_peer_grace_secs: 0.0,
                probe_timeout_secs: 0.1,
                probe_success_threshold_secs: 0.1,
                jail_ladder_secs: [1; 6],
            },
            QoSConfig {
                normal_threshold_secs: 600.0,
                deep_thinking_threshold_secs: 600.0,
                recovery_threshold_secs: 599.0,
                no_peer_grace_secs: 600.0,
                probe_timeout_secs: 120.0,
                probe_success_threshold_secs: 120.0,
                jail_ladder_secs: [86400; 6],
            },
        ];
        for config in configs {
            assert_eq!(config.validate(), Ok(()));
        }
    }

    #[test]
    fn deserialization_rejects_conflicting_thresholds() {
        for (override_fields, field) in [
            (
                json!({"recovery_threshold_secs": 20}),
                "recovery_threshold_secs",
            ),
            (
                json!({"recovery_threshold_secs": 21}),
                "recovery_threshold_secs",
            ),
            (
                json!({"deep_thinking_threshold_secs": 19}),
                "deep_thinking_threshold_secs",
            ),
            (
                json!({"probe_success_threshold_secs": 9}),
                "probe_success_threshold_secs",
            ),
        ] {
            let error = serde_json::from_value::<QoSConfig>(override_fields).unwrap_err();
            assert!(error.to_string().contains(field), "{error}");
        }
    }

    #[test]
    fn deserialization_rejects_invalid_ladders() {
        for ladder in [
            json!([]),
            json!([1, 2, 3, 4, 5]),
            json!([1, 2, 3, 4, 5, 6, 7]),
            json!([0, 2, 3, 4, 5, 6]),
            json!([1, 2, 3, 4, 5, 86401]),
            json!([1, 2, 3, 4, 6, 5]),
            json!([-1, 2, 3, 4, 5, 6]),
            json!([1.5, 2, 3, 4, 5, 6]),
        ] {
            assert!(
                serde_json::from_value::<QoSConfig>(json!({"jail_ladder_secs": ladder})).is_err(),
                "invalid ladder accepted: {ladder}"
            );
        }
    }

    #[test]
    fn explicit_invalid_fields_do_not_fall_back_to_defaults() {
        for fields in [
            json!({"normal_threshold_secs": null}),
            json!({"probe_timeout_secs": "8"}),
            json!({"no_peer_grace_secs": -1}),
            json!({"probe_timeout_secs": 121}),
            json!({"jail_ladder_secs": null}),
        ] {
            assert!(
                serde_json::from_value::<QoSConfig>(fields.clone()).is_err(),
                "{fields}"
            );
        }
    }
}
