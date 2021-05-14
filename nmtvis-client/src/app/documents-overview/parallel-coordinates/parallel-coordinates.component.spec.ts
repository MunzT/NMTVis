import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { ParallelCoordinatesComponent } from './parallel-coordinates.component';

describe('ParallelCoordinatesComponent', () => {
  let component: ParallelCoordinatesComponent;
  let fixture: ComponentFixture<ParallelCoordinatesComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ ParallelCoordinatesComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(ParallelCoordinatesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
