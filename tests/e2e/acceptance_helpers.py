"""工程验收共用的跨 HTTP 快照判错，不改变产品行为。"""
import httpx2


def confirmed_display_error(client: httpx2.Client, run_id: str, display: dict,
                            viewers: tuple[str, ...]) -> str | None:
    errors = [f'{viewer}: {display[viewer]["error"]}' for viewer in viewers
              if display[viewer]['status'] == 'error']
    if not errors:
        return None
    response = client.get(f'/runs/{run_id}')
    assert response.status_code == 200, response.text
    fresh = response.json()
    if fresh['state'] in ('COMPLETED', 'CANCELLED', 'FAILED', 'LOST') or fresh['worker_terminal'] is not None:
        return None
    return '; '.join(errors)
