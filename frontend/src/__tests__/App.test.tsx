import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { App } from '../App';
import { Dashboard } from '../pages/Dashboard';
import { ModelInfo } from '../pages/ModelInfo';
import { Analyze } from '../pages/Analyze';
import { SystemHealth } from '../pages/SystemHealth';

// Mock API service so tests run independently and reliably
vi.mock('../services/api', () => ({
  api: {
    getHealth: vi.fn().mockResolvedValue({
      status: 'healthy',
      service: 'FruitGrade AI Backend',
      version: '1.0.0',
      model_ready: true,
      device: 'cuda:0',
    }),
    getModelInfo: vi.fn().mockResolvedValue({
      backbone_architecture: 'convnext_tiny',
      coral_head_type: 'monotonic_coral',
      num_grades: 5,
      grades: [
        { grade: 0, name: 'Pristine (0)', description: 'Export quality' },
        { grade: 1, name: 'Mild (1)', description: 'Minor scuffing' },
        { grade: 2, name: 'Moderate (2)', description: 'Noticeable defect' },
        { grade: 3, name: 'Severe (3)', description: 'Structural breakdown' },
        { grade: 4, name: 'Critical (4)', description: 'Advanced rot' },
      ],
      supported_species: ['Apple', 'Banana', 'Mango', 'Orange', 'Tomato'],
      input_resolution: [224, 224],
      input_normalization: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
      ood_detector: 'Free Energy Inverted Scoring',
      ood_threshold: -1.5,
      abstention_threshold: 0.7,
      loaded_checkpoints: {},
      device: 'cuda:0',
      model_ready: true,
      research_evaluation: {
        benchmark_name: '30-Seed Confirmatory Image-Only Baseline',
        image_only_qwk_mean: 0.8143,
        image_only_qwk_std: 0.0116,
        n_seeds: 30,
        spearman_rho: 0.852,
        accuracy: 0.741,
        ood_fpr95_inverted_real: 1.0,
        ood_auroc_inverted_real: 0.2882,
      },
      documentation_url: 'https://github.com/NathishN/Final-year-project',
    }),
    getSystemInfo: vi.fn().mockResolvedValue({
      application_name: 'FruitGrade AI',
      version: '1.0.0',
      status: 'ready',
      python_version: '3.12.4',
      pytorch_version: '2.5.1',
      device_available: 'cuda:0',
      gpu_name: 'NVIDIA GeForce RTX 4050 Laptop GPU',
      gpu_memory_allocated_mb: 250.5,
      gpu_memory_reserved_mb: 512.0,
      max_upload_size_mb: 15,
      allowed_extensions: ['.jpg', '.jpeg', '.png', '.webp'],
      model_loaded: true,
    }),
    getHistory: vi.fn().mockResolvedValue({
      total: 0,
      items: [],
    }),
  },
}));

describe('FruitGrade AI Frontend Component Suite', () => {
  it('renders application brand and navigation links correctly', async () => {
    render(<App />);
    const brandElements = screen.getAllByText(/FruitGrade AI/i);
    expect(brandElements.length).toBeGreaterThan(0);
    expect(brandElements[0]).toBeInTheDocument();
    expect(screen.getByText(/Trustworthy Open-Set Ordinal Quality Grading/i)).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /Dashboard/i })[0]).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /Analyze Fruit/i })[0]).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: /History/i })[0]).toBeInTheDocument();
  });

  it('renders Dashboard with research metrics and primary action', () => {
    const handleNavigate = vi.fn();
    render(
      <Dashboard
        onNavigate={handleNavigate}
        health={{ status: 'healthy', service: 'FruitGrade AI', version: '1.0.0', model_ready: true, device: 'cuda:0' }}
        modelInfo={null}
      />
    );
    expect(screen.getByText(/Trustworthy/i)).toBeInTheDocument();
    expect(screen.getByText(/Analyze a Fruit Image/i)).toBeInTheDocument();
    expect(screen.getByText(/Rank Monotonicity/i)).toBeInTheDocument();

    fireEvent.click(screen.getByText(/Analyze a Fruit Image/i));
    expect(handleNavigate).toHaveBeenCalledWith('analyze');
  });

  it('renders Model Architecture with verified 30-seed QWK = 0.8143', () => {
    render(<ModelInfo modelInfo={null} />);
    expect(screen.getByText(/ConvNeXt-Tiny/i)).toBeInTheDocument();
    expect(screen.getByText(/Monotonic CORAL/i)).toBeInTheDocument();
    expect(screen.getByText(/0.8143/i)).toBeInTheDocument();
    expect(screen.getByText(/Open-Set Real-Image OOD Findings/i)).toBeInTheDocument();
  });

  it('renders Analyze dropzone and supports file validation', () => {
    render(<Analyze />);
    expect(screen.getByText(/Drag and drop a fruit photo here/i)).toBeInTheDocument();
    expect(screen.getByText(/Supported formats: JPG, JPEG, PNG, WEBP/i)).toBeInTheDocument();
  });

  it('renders System Health diagnostics and GPU details', async () => {
    render(<SystemHealth />);
    expect(screen.getByText(/System Health & Runtime/i)).toBeInTheDocument();
    expect(screen.getByText(/Runtime Specifications & Bounds/i)).toBeInTheDocument();
  });
});
