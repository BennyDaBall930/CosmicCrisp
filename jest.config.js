const path = require('node:path');

/** @type {import('jest').Config} */
const config = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests/js', '<rootDir>/tests/prompts/harness'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'json'],
  moduleNameMapper: {
    '^/js/(.*)$': '<rootDir>/webui/js/$1',
    '^/components/(.*)$': '<rootDir>/webui/components/$1',
  },
  transform: {
    '^.+\.(ts|tsx)$': ['ts-jest', { tsconfig: '<rootDir>/tsconfig.json' }],
    '^.+\.(js|jsx)$': [
      'babel-jest',
      { configFile: path.resolve(__dirname, 'babel.config.cjs') },
    ],
  },
  collectCoverage: true,
  collectCoverageFrom: [
    '<rootDir>/js/**/*.ts',
    '<rootDir>/js/**/*.js',
    '<rootDir>/tests/prompts/harness/**/*.ts',
    '!**/__fixtures__/**',
    '!**/__mocks__/**',
    '!**/__snapshots__/**',
    '!**/node_modules/**',
  ],
  coverageDirectory: '<rootDir>/reports/coverage/js',
  coverageReporters: ['lcov', 'json-summary', 'text-summary'],
  reporters: [
    'default',
    [
      'jest-junit',
      {
        outputDirectory: 'reports/junit',
        outputName: 'js-tests.xml',
        addFileAttribute: 'true',
      },
    ],
  ],
  coverageThreshold: {
    global: {
      branches: 70,
      functions: 80,
      lines: 85,
      statements: 85,
    },
  },
  modulePathIgnorePatterns: [
    '<rootDir>/.qa-venv',
    '<rootDir>/venv',
    '<rootDir>/tmp',
    '<rootDir>/node_modules',
  ],
  watchPathIgnorePatterns: ['<rootDir>/venv/', '<rootDir>/node_modules/'],
  testPathIgnorePatterns: ['<rootDir>/node_modules/', '<rootDir>/.qa-venv/'],
  setupFilesAfterEnv: ['<rootDir>/tests/js/setupJest.ts'],
  maxWorkers: 4,
};

module.exports = config;
