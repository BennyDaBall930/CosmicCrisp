import { describe, expect, test } from '@jest/globals';
import * as fc from 'fast-check';

import {
  formatDateTime,
  getCurrentUTCISOString,
  getUserTimezone,
  toLocalTime,
  toUTCISOString,
} from '../../../webui/js/time-utils.js';

const SAMPLE_ISO = '2024-10-27T15:04:05.000Z';

describe('time-utils', () => {
  test('toLocalTime returns empty string for falsy input', () => {
    expect(toLocalTime('')).toBe('');
    expect(toLocalTime(undefined as unknown as string)).toBe('');
  });

  test('formatDateTime uses sensible defaults', () => {
    const formatted = formatDateTime(SAMPLE_ISO);
    expect(formatted).toContain('2024');
    expect(formatted).toMatch(/\d{1,2}/);
  });

  test('formatDateTime respects named presets', () => {
    const dateOnly = formatDateTime(SAMPLE_ISO, 'date');
    expect(dateOnly).toBe(toLocalTime(SAMPLE_ISO, { dateStyle: 'medium' }));
    const timeOnly = formatDateTime(SAMPLE_ISO, 'time');
    expect(timeOnly).toBe(toLocalTime(SAMPLE_ISO, { timeStyle: 'medium' }));
  });

  test('getCurrentUTCISOString returns ISO 8601 string', () => {
    const value = getCurrentUTCISOString();
    expect(() => new Date(value)).not.toThrow();
    expect(value.endsWith('Z')).toBe(true);
  });

  test('getUserTimezone exposes Intl timezone', () => {
    const tz = getUserTimezone();
    expect(typeof tz).toBe('string');
    expect(tz.length).toBeGreaterThan(0);
    expect(() => Intl.DateTimeFormat(undefined, { timeZone: tz })).not.toThrow();
  });

  test('toUTCISOString matches native implementation (property)', () => {
    fc.assert(
      fc.property(fc.date(), (date) => {
        expect(isFinite(date.getTime())).toBe(true);
        expect(toUTCISOString(date)).toBe(date.toISOString());
      }),
      { numRuns: 50 }
    );
  });

  test('format presets mirror toLocalTime options (property)', () => {
    const optionsMap = {
      full: { dateStyle: 'medium', timeStyle: 'medium' } as const,
      date: { dateStyle: 'medium' } as const,
      time: { timeStyle: 'medium' } as const,
      short: { dateStyle: 'short', timeStyle: 'short' } as const,
    };

    fc.assert(
      fc.property(fc.date(), fc.constantFrom<'full' | 'date' | 'time' | 'short'>('full', 'date', 'time', 'short'), (date, format) => {
        const iso = date.toISOString();
        const expected = toLocalTime(iso, optionsMap[format]);
        expect(formatDateTime(iso, format)).toBe(expected);
      }),
      { numRuns: 40 }
    );
  });
});
