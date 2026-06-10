//! One LoRA adapter spec parsed from the `--lora-modules` CLI flag.
//!
//! Two formats are accepted per entry, matching the Python frontend's
//! `LoRAParserAction`:
//!
//! - **Old**: `name=path` (must contain `=` and not contain `,`).
//! - **New**: JSON object `{"name": ..., "path": ..., "base_model_name": ...,
//!   "is_3d_lora_weight": ...}`.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_with::{DeserializeFromStr, SerializeDisplay};

/// One static LoRA adapter spec resolved from a single `--lora-modules` value.
#[derive(Debug, Clone, PartialEq, Eq, DeserializeFromStr, SerializeDisplay)]
pub struct LoraModuleSpec {
    pub name: String,
    pub path: String,
    pub base_model_name: Option<String>,
    pub is_3d_lora_weight: bool,
}

/// Errors produced while parsing a single `--lora-modules` value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraModuleSpecError {
    Empty,
    OldFormatMissingName,
    OldFormatMissingPath,
    JsonInvalid { message: String },
    JsonMissingField { field: &'static str },
}

impl fmt::Display for LoraModuleSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("--lora-modules value cannot be empty"),
            Self::OldFormatMissingName => {
                f.write_str("--lora-modules value `name=path` is missing the name")
            }
            Self::OldFormatMissingPath => {
                f.write_str("--lora-modules value `name=path` is missing the path")
            }
            Self::JsonInvalid { message } => {
                write!(f, "--lora-modules value is not valid JSON: {message}")
            }
            Self::JsonMissingField { field } => {
                write!(
                    f,
                    "--lora-modules JSON value is missing required field `{field}`"
                )
            }
        }
    }
}

impl std::error::Error for LoraModuleSpecError {}

/// JSON shape mirroring Python's `LoRAModulePath` dataclass.
#[derive(Debug, Deserialize, Serialize)]
struct LoraModuleJson {
    name: Option<String>,
    path: Option<String>,
    #[serde(default)]
    base_model_name: Option<String>,
    #[serde(default)]
    is_3d_lora_weight: bool,
}

impl FromStr for LoraModuleSpec {
    type Err = LoraModuleSpecError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.is_empty() {
            return Err(LoraModuleSpecError::Empty);
        }

        // Old format `name=path` is detected by the same heuristic Python uses:
        // contains `=` and does not contain `,`. Otherwise treat as JSON.
        if value.contains('=') && !value.contains(',') {
            let (name, path) = value.split_once('=').expect("checked contains('=')");
            if name.is_empty() {
                return Err(LoraModuleSpecError::OldFormatMissingName);
            }
            if path.is_empty() {
                return Err(LoraModuleSpecError::OldFormatMissingPath);
            }
            return Ok(Self {
                name: name.to_string(),
                path: path.to_string(),
                base_model_name: None,
                is_3d_lora_weight: false,
            });
        }

        let parsed: LoraModuleJson =
            serde_json::from_str(value).map_err(|error| LoraModuleSpecError::JsonInvalid {
                message: error.to_string(),
            })?;
        let name = parsed.name.ok_or(LoraModuleSpecError::JsonMissingField { field: "name" })?;
        let path = parsed.path.ok_or(LoraModuleSpecError::JsonMissingField { field: "path" })?;
        Ok(Self {
            name,
            path,
            base_model_name: parsed.base_model_name,
            is_3d_lora_weight: parsed.is_3d_lora_weight,
        })
    }
}

impl fmt::Display for LoraModuleSpec {
    /// Render in the new JSON form so a `LoraModuleSpec` can roundtrip through
    /// the serde representation used by `Config` serialization.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let json = LoraModuleJson {
            name: Some(self.name.clone()),
            path: Some(self.path.clone()),
            base_model_name: self.base_model_name.clone(),
            is_3d_lora_weight: self.is_3d_lora_weight,
        };
        let rendered = serde_json::to_string(&json).map_err(|_| fmt::Error)?;
        f.write_str(&rendered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_old_format() {
        let spec: LoraModuleSpec = "sql-lora=/models/sql-lora".parse().unwrap();
        assert_eq!(
            spec,
            LoraModuleSpec {
                name: "sql-lora".to_string(),
                path: "/models/sql-lora".to_string(),
                base_model_name: None,
                is_3d_lora_weight: false,
            }
        );
    }

    #[test]
    fn parses_minimal_json() {
        let spec: LoraModuleSpec =
            r#"{"name":"sql-lora","path":"/models/sql-lora"}"#.parse().unwrap();
        assert_eq!(
            spec,
            LoraModuleSpec {
                name: "sql-lora".to_string(),
                path: "/models/sql-lora".to_string(),
                base_model_name: None,
                is_3d_lora_weight: false,
            }
        );
    }

    #[test]
    fn parses_full_json_with_base_model_and_3d_flag() {
        let value = r#"{
            "name": "sql-lora",
            "path": "/models/sql-lora",
            "base_model_name": "Qwen/Qwen3-0.6B",
            "is_3d_lora_weight": true
        }"#;
        let spec: LoraModuleSpec = value.parse().unwrap();
        assert_eq!(
            spec,
            LoraModuleSpec {
                name: "sql-lora".to_string(),
                path: "/models/sql-lora".to_string(),
                base_model_name: Some("Qwen/Qwen3-0.6B".to_string()),
                is_3d_lora_weight: true,
            }
        );
    }

    #[test]
    fn old_format_wins_when_value_has_equals_and_no_comma() {
        // The Python heuristic also routes `name=path` to the old format even
        // when the path superficially looks like it could be JSON.
        let spec: LoraModuleSpec = "name={path".parse().unwrap();
        assert_eq!(spec.name, "name");
        assert_eq!(spec.path, "{path");
    }

    #[test]
    fn json_path_taken_when_value_contains_comma() {
        let value = r#"{"name":"sql-lora","path":"/models/sql-lora"}"#;
        let spec: LoraModuleSpec = value.parse().unwrap();
        assert_eq!(spec.name, "sql-lora");
    }

    #[test]
    fn rejects_empty_string() {
        assert_eq!(
            "".parse::<LoraModuleSpec>().unwrap_err(),
            LoraModuleSpecError::Empty
        );
    }

    #[test]
    fn rejects_old_format_missing_name() {
        assert_eq!(
            "=/models/sql".parse::<LoraModuleSpec>().unwrap_err(),
            LoraModuleSpecError::OldFormatMissingName
        );
    }

    #[test]
    fn rejects_old_format_missing_path() {
        assert_eq!(
            "sql-lora=".parse::<LoraModuleSpec>().unwrap_err(),
            LoraModuleSpecError::OldFormatMissingPath
        );
    }

    #[test]
    fn rejects_malformed_json() {
        let error = r#"{"name": "sql-lora", "path"#.parse::<LoraModuleSpec>().unwrap_err();
        assert!(matches!(error, LoraModuleSpecError::JsonInvalid { .. }));
    }

    #[test]
    fn rejects_json_missing_name() {
        assert_eq!(
            r#"{"path":"/models/sql-lora"}"#.parse::<LoraModuleSpec>().unwrap_err(),
            LoraModuleSpecError::JsonMissingField { field: "name" }
        );
    }

    #[test]
    fn rejects_json_missing_path() {
        assert_eq!(
            r#"{"name":"sql-lora"}"#.parse::<LoraModuleSpec>().unwrap_err(),
            LoraModuleSpecError::JsonMissingField { field: "path" }
        );
    }

    #[test]
    fn displays_as_json_for_serde_roundtrip() {
        let spec = LoraModuleSpec {
            name: "sql-lora".to_string(),
            path: "/models/sql-lora".to_string(),
            base_model_name: Some("Qwen/Qwen3-0.6B".to_string()),
            is_3d_lora_weight: true,
        };
        let rendered = spec.to_string();
        let parsed: LoraModuleSpec = rendered.parse().unwrap();
        assert_eq!(parsed, spec);
    }
}
