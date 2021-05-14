import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { DocumentsOverviewComponent } from './documents-overview.component';

describe('DocumentsOverviewComponent', () => {
  let component: DocumentsOverviewComponent;
  let fixture: ComponentFixture<DocumentsOverviewComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ DocumentsOverviewComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(DocumentsOverviewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
