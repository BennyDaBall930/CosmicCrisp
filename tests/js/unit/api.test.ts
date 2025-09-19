import { afterEach, beforeEach, describe, expect, test } from '@jest/globals';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let documentBackup: any;

describe('webui api client', () => {
  beforeEach(() => {
    jest.resetModules();
    documentBackup = global.document;
    global.document = { cookie: '' } as unknown as Document;
  });

  afterEach(() => {
    global.document = documentBackup;
    jest.restoreAllMocks();
  });

  test('fetchApi injects CSRF token and caches it', async () => {
    const fetchMock = jest.fn(async (url: RequestInfo, init?: RequestInit) => {
      if (typeof url === 'string' && url.includes('csrf_token')) {
        return {
          json: async () => ({ token: 'abc123', runtime_id: 'r-local' }),
        } as Response;
      }
      return {
        status: 200,
        ok: true,
        json: async () => ({ ok: true }),
      } as Response;
    });

    // @ts-expect-error allow assignment for test double
    global.fetch = fetchMock;

    const { fetchApi } = await import('../../../webui/js/api.js');

    const response = await fetchApi('/ping', { method: 'POST' });
    expect(response.ok).toBe(true);

    expect(fetchMock).toHaveBeenCalledWith('/csrf_token', expect.any(Object));
    expect(fetchMock).toHaveBeenLastCalledWith('/ping', expect.objectContaining({
      headers: expect.objectContaining({ 'X-CSRF-Token': 'abc123' }),
      method: 'POST',
    }));

    // second call should reuse cached token without hitting csrf endpoint again
    await fetchApi('/ping', { method: 'POST' });
    const csrfCalls = fetchMock.mock.calls.filter(([url]) => typeof url === 'string' && url.includes('csrf_token'));
    expect(csrfCalls).toHaveLength(1);
  });

  test('fetchApi retries once on 403 and forces token refresh', async () => {
    let token = 'stale';
    const headerHistory: string[] = [];
    const fetchMock = jest.fn(async (url: RequestInfo, init?: RequestInit) => {
      if (typeof url === 'string' && url.includes('csrf_token')) {
        token = token === 'stale' ? 'fresh-1' : 'fresh-2';
        return { json: async () => ({ token, runtime_id: 'r-local' }) } as Response;
      }
      if (typeof url === 'string' && url === '/secure') {
        const headerValue = (init?.headers as Record<string, string>)?.['X-CSRF-Token'];
        headerHistory.push(headerValue ?? '');
        if (headerValue === 'fresh-1') {
          return { status: 403, ok: false, text: async () => 'Forbidden' } as Response;
        }
      }
      return { status: 200, ok: true, json: async () => ({ ok: true }) } as Response;
    });

    // @ts-expect-error allow assignment for test double
    global.fetch = fetchMock;

    const { fetchApi } = await import('../../../webui/js/api.js');
    const res = await fetchApi('/secure', { method: 'POST' });
    expect(res.ok).toBe(true);

    const pingCalls = fetchMock.mock.calls.filter(([url]) => typeof url === 'string' && url === '/secure');
    expect(pingCalls).toHaveLength(2);
    expect(headerHistory).toEqual(['fresh-1', 'fresh-2']);
    const csrfCalls = fetchMock.mock.calls.filter(([url]) => typeof url === 'string' && url.includes('csrf_token'));
    expect(csrfCalls).toHaveLength(2);
  });

  test('callJsonApi serialises payload and surfaces http errors', async () => {
    const fetchMock = jest.fn(async (url: RequestInfo) => {
      if (url === '/csrf_token') {
        return { json: async () => ({ token: 'json-1', runtime_id: 'r-local' }) } as Response;
      }
      return {
        status: 400,
        ok: false,
        text: async () => 'Bad request',
      } as Response;
    });

    // @ts-expect-error allow assignment for test double
    global.fetch = fetchMock;

    const { callJsonApi } = await import('../../../webui/js/api.js');

    await expect(callJsonApi('/endpoint', { foo: 'bar' })).rejects.toThrow('Bad request');

    const payloadCall = fetchMock.mock.calls.find(([url]) => url === '/endpoint') as ([RequestInfo, RequestInit] | undefined);
    expect(payloadCall).toBeDefined();
    const [, init] = payloadCall!;
    expect(init?.body).toBe(JSON.stringify({ foo: 'bar' }));
    expect(init?.headers).toMatchObject({ 'Content-Type': 'application/json', 'X-CSRF-Token': 'json-1' });
  });
});
