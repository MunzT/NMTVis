import { TestBed, inject } from '@angular/core/testing';

import { ExperimentService } from './experiment.service';

describe('ExperimentService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [ExperimentService]
    });
  });

  it('should be created', inject([ExperimentService], (service: ExperimentService) => {
    expect(service).toBeTruthy();
  }));
});
