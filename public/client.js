// In production, this should check CSRF, and not pass the session ID.
// The customer ID for the portal should be pulled from the
// authenticated user on the server.
document.addEventListener('DOMContentLoaded', async () => {
  let searchParams = new URLSearchParams(window.location.search);
  if (searchParams.has('session_id')) {
    const session_id = searchParams.get('session_id');
    const sessionIdInput = document.getElementById('session-id');
    if (sessionIdInput) {
      sessionIdInput.setAttribute('value', session_id);
    }
  }
});

