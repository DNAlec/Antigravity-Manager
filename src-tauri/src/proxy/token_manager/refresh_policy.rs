//! OAuth refresh failures must not silently migrate a warm text session.

use std::fmt;

#[derive(Debug)]
pub(super) enum TokenRefreshError {
    Temporary(String),
    CredentialRejected(String),
}

impl TokenRefreshError {
    pub(super) fn from_oauth(message: String) -> Self {
        // The OAuth module returns strings after trying configured client IDs.
        // Only an explicit OAuth rejection establishes that another account is
        // needed. Timeouts, transport failures and malformed server responses
        // do not establish that the bound account's credentials are invalid.
        let lower = message.to_ascii_lowercase();
        if [
            "invalid_grant",
            "invalid_client",
            "unauthorized_client",
            "access_denied",
        ]
        .iter()
        .any(|code| lower.contains(code))
        {
            Self::CredentialRejected(message)
        } else {
            Self::Temporary(message)
        }
    }

    pub(super) fn is_temporary(&self) -> bool {
        matches!(self, Self::Temporary(_))
    }
}

impl fmt::Display for TokenRefreshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Temporary(message) | Self::CredentialRejected(message) => message.fmt(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oauth_transient_failures_do_not_invalidate_credentials() {
        for message in [
            "Refresh request failed: connection reset by peer",
            "Refresh data parsing failed: unexpected EOF",
            "Refresh failed for client [default]: {\"error\":\"temporarily_unavailable\"}",
            "Refresh failed: upstream HTTP 503",
        ] {
            assert!(TokenRefreshError::from_oauth(message.into()).is_temporary());
        }
    }

    #[test]
    fn explicit_oauth_rejections_retain_account_recovery() {
        for code in [
            "invalid_grant",
            "invalid_client",
            "unauthorized_client",
            "access_denied",
        ] {
            let message = format!("Refresh failed for client [default]: {{\"error\":\"{code}\"}}");
            let failure = TokenRefreshError::from_oauth(message.clone());
            assert!(!failure.is_temporary());
            assert_eq!(failure.to_string(), message);
        }
    }
}
